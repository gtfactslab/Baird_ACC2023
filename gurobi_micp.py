from ..base import STLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np

import gurobipy as gp
from gurobipy import GRB

import time

class GurobiMICPSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \min & -\\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = A x_t + B u_t

        & y_{t} = C x_t + D u_t

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    with Gurobi using mixed-integer convex programming. This gives a globally optimal
    solution, but may be computationally expensive for long and complex specifications.
    
    .. note::

        This class implements the algorithm described in

        Belta C, et al.
        *Formal methods for control synthesis: an optimization perspective*.
        Annual Review of Control, Robotics, and Autonomous Systems, 2019.
        https://dx.doi.org/10.1146/annurev-control-053018-023717.

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    :param presolve:        (optional) A boolean indicating whether to use Gurobi's
                            presolve routines. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    """

    def __init__(self, spec, sys, x0, T, M=1000, robustness_cost=True, 
            presolve=True, verbose=True, horizon=0):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T, verbose)

        self.horizon = horizon # useful for our specific problem L.B.

        self.M = float(M)
        self.presolve = presolve

        # Set up the optimization problem
        self.model = gp.Model("STL_MICP")
        
        # Store the cost function, which will added to self.model right before solving
        self.cost = 0.0

        # Store a starting useful point for dynamic constraints
        self.start_point = max(0, x0.shape[1] - self.horizon - 1)

        # Set some model parameters
        if not self.presolve:
            self.model.setParam('Presolve', 0)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time
        # self.T+4 - self.horizon# 
        self.q = 1#self.T+1 - self.horizon - max(self.T - 2*self.horizon, 0)
        print(f'q: {self.q}')

        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.T), lb=-float('inf'), name='y')
        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='x')
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='u')
        self.s = self.model.addMVar(1, lb=-float('inf'), name='s')
        self.rho = self.model.addMVar((1, self.q), name="rho",lb=0.0) # lb sets minimum robustness

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()

        if self.verbose:
            print(f"Setup complete in {time.time()-st} seconds.")

    def AddControlBounds(self, u_min, u_max):
        for t in range(self.start_point, self.T):
            self.model.addConstr( u_min <= self.u[:,t] )
            self.model.addConstr( self.u[:,t] <= u_max )

    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T):
            self.model.addConstr( x_min <= self.x[:,t] )
            self.model.addConstr( self.x[:,t] <= x_max )

    def AddQuadraticCost(self, Q, R):
        self.cost += self.x[:,0]@Q@self.x[:,0] + self.u[:,0]@R@self.u[:,0]
        for t in range(1,self.T):
            self.cost += self.x[:,t]@Q@self.x[:,t] + self.u[:,t]@R@self.u[:,t]

        print(type(self.cost))

    def AddIthControlCost(self, u_hat, i, u_hat_greater=True):
        if u_hat_greater:
            for j in range(self.u.shape[0]):
                self.cost += u_hat[j] - self.u[j, i]
            self.model.addConstr( u_hat - self.u[:, i] >= 0)
        else:
            for j in range(self.u.shape[0]):
                self.cost += self.u[j, i] - u_hat[j]
            self.model.addConstr( self.u[:, i] - u_hat >= 0)

        # self.cost += (u_hat - self.u[:, i]) * (u_hat - self.u[:, i])
        print(type(self.cost))

        # print(self.cost)

    def AddLPCost(self):
        self.cost += self.s

    def AddLPConstraints(self, i, u_hat):
        # Add two constraints with matrices appropriately
        self.model.addConstr( self.u[:, i] - self.s <= u_hat )
        self.model.addConstr( -self.s - self.u[:, i] <= -u_hat )

    def AddControlCost(self, u_hat):
        for t in range(self.T):
            self.cost += (u_hat - self.u[:, t]) * (self.T - t)
    
    def AddRobustnessCost(self):
        self.cost -= 1*self.rho

    def AddRecursiveFeasibilityConstraint(self, polytope):
        # ensure H@self.y[:, i] <= h
        for i in range(polytope.p.shape[1]):
            self.model.addConstr( polytope.P[i,:]@self.x[:,-1] <= polytope.p[i] )

    def AddRobustnessConstraint(self, rho_min=0.0):
        for t in range(self.q):
            self.model.addConstr( self.rho[0, t] >= rho_min )
        # self.model.addConstr( self.rho >= rho_min )

    def Solve(self):
        # Set the cost function now, right before we solve.
        # This is needed since model.setObjective resets the cost.
        # print(type(self.cost))
        # print(self.cost)
        self.model.setObjective(self.cost, GRB.MINIMIZE)
        print(self.model.getVars())
        # Do the actual solving
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            rho = self.rho.X

            # Report optimal cost and robustness
            if self.verbose:
                print("Solve time: ", self.model.Runtime)
                print("Optimal robustness: ", rho)
                print("")
                # for i in range(len(self.z_specs)):
                #     print(f'Resultant z_specs[{i}]: ', self.z_specs[i].X)
        else:
            if self.verbose:
                print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            rho = -np.inf

        return (x,u,rho,self.model.Runtime,self.model.getObjective().getValue())

    def AddDynamicsConstraints(self):
        h = self.x0.shape[1]
        # Initial condition
        #self.model.addConstr( self.x[:,0] == self.x0 )

        # Update with historical conditions!
        # self.model.addConstr( self.x[:, 0:self.x0.shape[1]] == self.x0)

        # # Update with historical conditions!
        print('x0')
        print(self.x0)
        for i in range(self.start_point, h):
            print(f'initializing index {i}, x[:,{i}]')
            print(self.x0[:, i:i+1])
            self.model.addConstr( self.x[:, i] == self.x0[:, i] )

        # Dynamics
        for t in range(h-1, self.T-1):
            print(f'initializing index {t}, x[:,{t+1}], y[:,{t}] using u[:,{t}]')
            self.model.addConstr(
                    self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] )
        for t in range(self.T-1):
            self.model.addConstr(
                    self.y[:,t] == self.sys.C@self.x[:,t] ) # + self.sys.D@self.u[:,t] )
        print(f'initializing index {self.T-1}, y only')
        self.model.addConstr(
                self.y[:,self.T-1] == self.sys.C@self.x[:,self.T-1] ) # + self.sys.D@self.u[:,self.T-1] )


    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value

        #
        self.z_spec = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
        start = max(self.x0.shape[1] - self.horizon, 0)
        temp_spec = self.spec.always(start, self.x0.shape[1] + 1)
        print(f'range is {start} to {self.x0.shape[1] + 1}')
        self.AddSubformulaConstraints(temp_spec, self.z_spec, 0, 0)
        self.model.addConstr( self.z_spec == 1)
        return

        idx=0
        self.z_spec = self.model.addMVar(1, vtype=GRB.CONTINUOUS)#[None] * self.q # + 1
        # Construct this as AND each of the indices.
        for t in range(max(self.T - 2*self.horizon, 0), self.T - self.horizon + 1): 
            print(f'adding STL constraint to index {t}')
            print(f'idx == {idx}')
            #self.z_specs[idx] = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
            if self.verbose:
                print(self.AddSubformulaConstraints(self.spec, self.z_spec, t, idx)) # s[idx]
            else:
                self.AddSubformulaConstraints(self.spec, self.z_spec, t, idx)
            # self.model.addConstr( self.z_specs[idx] == 1 )
            idx+=1
        self.model.addConstr( self.z_spec == 1 )

    def AddSubformulaConstraints(self, formula, z, t, idx):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t).

        If the formula is a predicate, this constraint uses the "big-M"
        formulation

            A[x(t);u(t)] - b + (1-z)M >= 0,

        which enforces A[x;u] - b >= 0 if z=1, where (A,b) are the
        linear constraints associated with this predicate.

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold).
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            self.model.addConstr( formula.a.T@self.y[:,t] - formula.b + (1-z)*self.M  >= self.rho[0,idx] )

            # Force z to be binary
            b = self.model.addMVar(1,vtype=GRB.BINARY)
            self.model.addConstr(z == b)

            return f'A@y[:,{t}] - b  >= rho[{idx}]'
        
        elif isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            returnString = ""
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    if self.verbose:
                        pass
                        # print(f'timestep: {formula.timesteps[i]}, i: {i}')
                        # print(subformula)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula
                                                   # should hold
                    returnString += " AND (" + self.AddSubformulaConstraints(subformula, z_sub, t+t_sub, idx) + ")"
                    self.model.addConstr( z <= z_sub )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    if self.verbose:
                        pass
                        # print(f'timestep: {formula.timesteps[i]}, i: {i}')
                        # print(subformula)
                    z_subs.append(z_sub)
                    t_sub = formula.timesteps[i]
                    returnString += " OR (" + self.AddSubformulaConstraints(subformula, z_sub, t+t_sub, idx) + ")"
                self.model.addConstr( z <= sum(z_subs) )

            return returnString

