from ..base import STLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np
from scipy.ndimage import shift

import gurobipy as gp
from gurobipy import GRB #, QuadExpr

import time

class GurobiMICPWarmStartSolver(STLSolver):
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

    :param spec:            A tuple of :class:`.STLFormula` describing the specification, logically conjoined.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param model:           A :class:`Model` describing the system dynamics as an LTV system.
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

    def __init__(self, spec, sys, x0, T, infinite_spec=None, model=None, M=1000, robustness_cost=False, 
            hard_constraint=True, presolve=True, verbose=True, horizon=0, history=0, dT=0.25, rho_min=0.0, N=1):
        assert M > 0, "M should be a (large) positive scalar"
        if type(spec) is not tuple:
            spec = (spec,)
        super().__init__(spec, sys, x0, T, verbose)

        # self.dT=dT

        self.horizon = horizon # useful for our specific problem
        self.dynamics_model = model

        self.M = float(M)
        self.presolve = presolve

        # Create a variable representing the number of future steps to require rho>0 on.
        self.N = N
        

        # Create a gurobi model to govern the optimization problem.
        self.model = gp.Model("STL_MICP")
        
        # Initialize the cost function, which will added to self.model right before solving
        self.cost = 0.0

        # Initialize a place to hold constraints
        self.dynamics_constraints = []
        self.lp_constraints = []
        self.infinite_spec_constraints = []

        self.initialization_point = None

        # Dummy start point - it's not that useful...
        #self.start_point = 2*self.horizon
        self.start_writing_ics = 0

        # Set some model parameters
        if not self.presolve:
            self.model.setParam('Presolve', 0)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time
        
        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.T), lb=-float('inf'), name='y')
        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='x')
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='u')
        self.s = self.model.addMVar((self.sys.m, 1), lb=-float('inf'), name='s')
        self.rho = self.model.addMVar((len(spec), 1, self.T), name='rho', lb=rho_min) # lb sets minimum robustness
        self.ξ = self.model.addMVar((len(spec), 1, self.T), name='ξ', lb=0.0) # slack variable for the robustness

        # Create the Pre vector.
        self.infinite_spec=None; self.Pre=None
        if infinite_spec is not None:
            self.infinite_spec = infinite_spec
            # self.Pre = np.ones((infinite_spec.delay, 1)) * np.inf # all infinite, and it will be updated recursively like a controller.
            # self.Pre = np.ones((history, 1)) * self.M
            self.Pre = self.model.addMVar((history, 1), lb=-float('inf'), name='Pre')
            self.alpha = np.ones((history, 1)) * self.M
            self.history = history
            self.pre_constraints = [] # for updating element-wise the upper bound on the Pre variables.
            # print(self.Pre)

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints() # Update: this is called only when first initializing the problem.
        self.AddSTLConstraints()
        if hard_constraint:
            self.AddRobustnessConstraint(rho_min=rho_min)
        if robustness_cost:
            self.AddRobustnessCost()
            self.AddSoftRobustnessConstraint()

        if self.verbose:
            print(f"Setup complete in {time.time()-st} seconds.")

    def updateModel(self, sys, x0, i, u_hat, initialization_point=None):
        # Updates the Gurobi model with new system information
        # (remember: we handle LTV systems this way)

        # i is the "current time step." As far as we are concerned for the model, we
        # want to start at (history - i) + horizon.
        self.start_writing_ics = max(0, self.horizon-i)
        # so, we will ideally write the last 1,2,...,self.horizon time steps of x0.
        # this is the index to start writing x0 into the model at.
        #max(self.horizon, 2*self.horizon - i)
        self.x0 = x0
        self.sys = sys

        if initialization_point is not None:
            self.initialization_point = initialization_point # used for the dynamics constraints with a NL model.

        self.RemoveDynamicsConstraints()
        # self.AddDynamicsConstraints()
        self.AddHistoricDynamicsConstraints()

        if self.infinite_spec is not None:
            self.RemoveInfiniteSpecConstraints()
            self.AddInfiniteSpecConstraints(self.horizon, self.z_spec)
        
        self.RemoveLPConstraints()
        self.AddLPConstraints(self.horizon-1, u_hat)

    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T): # range(self.start_point, self.T)
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

        if self.verbose:
            print(type(self.cost))

    def AddLPCost(self):
        self.cost += (np.ones((1, self.sys.m)) @ self.s)[0,0]

    def AddLPConstraints(self, i, u_hat):
        # Add two constraints with matrices appropriately
        self.lp_constraints.append( self.u[:, i:i+1] - self.s <= u_hat )
        self.lp_constraints.append( -self.s - self.u[:, i:i+1] <= -u_hat )

        self.lp_constraints = self.model.addConstrs(_ for _ in self.lp_constraints)

    def RemoveLPConstraints(self):
        if len(self.lp_constraints) > 0:
            self.model.remove(self.lp_constraints)
        self.lp_constraints = []
        return

    def AddControlCost(self, u_hat):
        for t in range(self.T):
            self.cost += (u_hat - self.u[:, t]) * (self.T - t)
    
    def AddRobustnessCost(self):
        for j in range(len(self.spec)):
            for t in range(self.T - self.horizon): # range(self.N + self.horizon):
                if j == 0:
                    gamma_t = 1000
                else:
                    gamma_t = 500
                if t < self.horizon:
                    gamma_t *= 1
                else:
                    pass
                    # gamma_t *= (0.5 ** (t - self.horizon))
                    # gamma_t = 0.5 ** (t - self.horizon)
                self.cost += gamma_t * self.ξ[j,0,t]
            # self.cost -= 1*self.rho

    def AddRecursiveFeasibilityConstraint(self, polytope):
        if type(polytope) is not tuple:
            for i in range(polytope.p.shape[0]): # number of constraints
                self.model.addConstr( polytope.P[i,:]@self.x[:, self.N + self.horizon + 1] <= polytope.p[i] )
        else:
            mainPolytope = polytope[0]
            for i in range(mainPolytope.p.shape[0]):
                self.model.addConstr( mainPolytope.P[i,:]@self.x[:, self.N + self.horizon + 1] <= mainPolytope.p[i] )
            
            # For the rest: conjoin these using integer variables.
            n = len(polytope) - 1 # number of polytopes to join.
            z = self.model.addMVar(n, vtype=GRB.BINARY)
            for j in range(n):
                for i in range(polytope[j+1].p.shape[0]):
                    self.model.addConstr(polytope[j+1].P[i,:]@self.x[:, self.N + self.horizon + 1] <= polytope[j+1].p[i] + self.M*(1-z[j]))
            self.model.addConstr(sum(z) == 1)
                

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.model.addConstr( self.rho >= rho_min )
    
    def AddSoftRobustnessConstraint(self):
        self.model.addConstr(self.rho >= -self.ξ)

    def Solve(self):
        # Set the cost function now, right before we solve.
        # This is needed since model.setObjective resets the cost.
        # print(type(self.cost))
        # print(self.cost)
        self.model.setObjective(self.cost, GRB.MINIMIZE)
        # print(self.model.getVars())
        # Do the actual solving
        self.model.optimize()
        success = None

        if self.model.status == GRB.OPTIMAL:
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            rho = self.rho.X

            # Update the Pre vector for the Since clause using its LinearPredicate formula.
            if self.Pre is not None:
                # Calculate the predicate value of the current time step. What is the current time?
                currentPredicateValue = (self.infinite_spec.a * self.y.X[:,self.start_writing_ics] - self.infinite_spec.b) # maybe self.horizon?
                updatePreValue = min(np.min(self.alpha), currentPredicateValue) # + 1e-6
                # if updatePreValue < 0:
                #     updatePreValue -= 1
                self.alpha = np.array([shift(self.alpha[:, 0], -1, cval=updatePreValue)]).T
                if self.verbose:
                    print(f'test: {self.infinite_spec.a * self.y.X[:,self.start_writing_ics] - self.infinite_spec.b}')
                    print(f'new pre = {updatePreValue}')
                    print(f'self.alpha = {self.alpha}')
                    if updatePreValue <= 0:
                        print(f'updatePreValue == {updatePreValue}, and now it\'s negative!\n\n\n this is bad.')
                self.UpdatePreConstraints()
                # time.sleep(1)

            # Report optimal cost and robustness
            if self.verbose:
                print("Solve time: ", self.model.Runtime)
                print("Optimal robustness: ", rho)
                print("")
                # for i in range(len(self.z_specs)):
                #     print(f'Resultant z_specs[{i}]: ', self.z_specs[i].X)
            
            success = True
            objective = self.model.getObjective().getValue()
        else:
            if self.verbose:
                print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            rho = -np.inf
            success = False
            objective = np.inf
        
        if self.verbose:
            print(self.model.status)
        return (x,u,rho,self.model.Runtime,objective,success)
    
    def AddHistoricDynamicsConstraints(self):
        h = self.x0.shape[1] # number of historical conditions.

        # Update with historical conditions
        if self.verbose:
            print('x0')
            print(self.x0)

        for i in range(self.start_writing_ics, self.horizon):
            x0_idx = h - self.horizon + i# indexer corresponding to the last values of x0
            if self.verbose:
                print(f'initializing index {i}, x[:,{i}]')
                print(self.x0[:, x0_idx:x0_idx+1])
            
            # Eliminate C array slicing due to Vars vs MVars.
            self.dynamics_constraints.append( self.x[:, i] == self.x0[:, x0_idx] )

        self.dynamics_constraints = self.model.addConstrs((_ for _ in self.dynamics_constraints))
        # There must exist a more efficient way of doing this.

    def AddDynamicsConstraints(self):
        # Dynamics (that are not updated with each update to the historical states)
        if self.dynamics_model is None:
            for t in range(self.horizon-1, self.T-1):
                if self.verbose:
                    print(f'initializing index {t}, x[:,{t+1}], y[:,{t}] using u[:,{t}]')
                self.model.addConstr(
                        self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] )
            for t in range(self.T):
                self.model.addConstr(
                        self.y[:,t] == self.sys.C@self.x[:,t] )
        else:
            if self.verbose:
                print('x0')
                print(self.x0[:, -1:])

            A_discrete, B_discrete = self.dynamics_model.get_discrete_model()
            
            for t in range(self.horizon-1, self.T-1):
                # indexer = t - (self.horizon - 1)
                # mI = t - (self.horizon-1)
                if self.verbose:
                    print(f'initializing index {t}, x[:,{t+1}] using u[:,{t}] and model')
                self.model.addConstr(
                        self.x[:,t+1] == A_discrete@self.x[:,t] + B_discrete@self.u[:,t] ) 
                
            for t in range(self.T):
                if self.verbose:
                    print(f'initializing index {t}, y[:,{t}] using y=C@x[:,{t}]')
                self.model.addConstr(
                        self.y[:,t] == self.dynamics_model.C@self.x[:,t] )

    def RemoveDynamicsConstraints(self):
        # remove previously added dynamics constraints.
        if len(self.dynamics_constraints) > 0:
            self.model.remove(self.dynamics_constraints)
        
        # reset the array
        self.dynamics_constraints = []

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
        for j in range(len(self.spec)):
            start = 0#max(self.x0.shape[1] - self.horizon, 0)
            temp_spec = self.spec[j].always(start, self.horizon+self.N)
            if self.verbose:
                print(f'range is {start} to {self.horizon + self.N}')
            self.AddFormulaConstraintsNotRecursive(temp_spec, self.z_spec, 0, 0, j)
            # self.AddPreConstraints(self.z_spec)
            # self.AddSubformulaConstraints(temp_spec, self.z_spec, 0, 0)

        if self.infinite_spec is not None:
            self.AddInfiniteSpecConstraints(self.horizon, self.z_spec)

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
            self.model.addConstr( formula.a*self.y[:,t] - formula.b + (1-z)*self.M  >= self.rho[0,t] )
            # A.T@B is the same as A*B
            # Force z to be binary
            b = self.model.addMVar(1,vtype=GRB.BINARY)
            self.model.addConstr(z == b)

            return f'A@y[:,{t}] - b  >= rho[{idx}]'
        
        elif isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            # returnString = ""
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    t_sub = formula.timesteps[i]   # the timestep at which this formula
                                                   # should hold
                    # returnString += " AND (" + self.AddSubformulaConstraints(subformula, z_sub, t+t_sub, idx) + ")"
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub, idx)
                    self.model.addConstr( z <= z_sub )

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    z_subs.append(z_sub)
                    t_sub = formula.timesteps[i]
                    # returnString += " OR (" + self.AddSubformulaConstraints(subformula, z_sub, t+t_sub, idx) + ")"
                    self.AddSubformulaConstraints(subformula, z_sub, t+t_sub, idx)
                self.model.addConstr( z <= sum(z_subs) )

            # return returnString
    
    def AddInfiniteSpecConstraints(self, tMax, topZ):
        return # do nothing.
        # Add the constraints for self.infinite_spec
        # For the moment, assuming that self.infinite_spec has a delay and two predicates:
        # self.infinite_spec.once, and self.infinite_spec.always
        
        # For each time step, take the minimum of the Pre vector and the robustness of once.
        # When this becomes negative, start enforcing...
        # preIdx = self.horizon - np.argmin(np.flip(self.Pre)) - 1
        # print(np.argmin(np.flip(self.Pre)))
        # print(self.horizon - np.argmin(np.flip(self.Pre)) - 1)
        # preIdx = np.argmin(self.Pre) # - 1

        # preIdx = self.horizon - np.argmin(self.Pre) - 1
        # preIdx = first index where the argument is less than 0.
        npwhere = np.where(self.Pre < 0)[0]
        if npwhere.size != 0:
            preIdx = self.horizon - np.where(self.Pre < 0)[0][0] - 1

        pre = np.min(self.Pre)
        for t in range(tMax):
            # Calculate the Pre value.
            # print(np.min(pre, self.infinite_spec.once.a * self.y[:, t] - self.infinite_spec.once.b))
            # pre = min(pre, self.infinite_spec.once.a * self.y[:, t] - self.infinite_spec.once.b)

            # Do this intelligently: enforce the following constraints only if one of the two hold:
            # 1) self.infinite_spec.once.a * self.y[:, t] - self.infinite_spec.once.b <= 0
            # 2) pre <= 0

            # One idea: if pre > 0, either self.infinite_spec.once.a * self.y[:, t] - self.infinite_spec.once.b > 0
            # OR: the following conditions hold. (Could be both, we don't care)
            # For or, require that topZ <= sum(z).

            z_ors = []
            if pre > 0:
                z = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                self.infinite_spec_constraints.append( -(self.infinite_spec.once.a * self.y[:, t] - self.infinite_spec.once.b) + (1-z)*self.M >= 0 )
                b = self.model.addMVar(1, vtype=GRB.BINARY)
                self.infinite_spec_constraints.append(z == b)
                z_ors.append(z)

                z_and = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
                z_ors.append(z_and)
                for i in range(max(0, t + self.infinite_spec.delay), tMax):
                    z = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    self.infinite_spec_constraints.append(self.infinite_spec.always.a * self.y[:,i] - self.infinite_spec.always.b + (1 - z) * self.M >= self.rho)
                    b = self.model.addMVar(1, vtype=GRB.BINARY)
                    self.infinite_spec_constraints.append(z == b)
                    self.infinite_spec_constraints.append(z_and <= z)
                self.infinite_spec_constraints.append( topZ <= sum(z_ors) )

            # otherwise, we start enforcing the following constraints anyways (if pre <= 0)            
            else: # and i < t + self.infinite_spec.delay:
                print(f'preIdx: {preIdx}') #  - preIdx
                print(self.Pre)
                for i in range(max(0, t + self.infinite_spec.delay - preIdx + self.horizon + 1), tMax): #  - preIdx
                    z = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                    self.infinite_spec_constraints.append(self.infinite_spec.always.a * self.y[:,i] - self.infinite_spec.always.b + (1 - z) * self.M >= self.rho)
                    b = self.model.addMVar(1, vtype=GRB.BINARY)
                    self.infinite_spec_constraints.append(z == b)

                    self.infinite_spec_constraints.append(topZ <= z) # incorporate the constraint into the overall robustness.
                break
            # else: continue
        
        self.infinite_spec_constraints = self.model.addConstrs((_ for _ in self.infinite_spec_constraints))
        # return returnString

    def RemoveInfiniteSpecConstraints(self):
        return
        # remove previously added dynamics constraints.
        if len(self.infinite_spec_constraints) > 0:
            self.model.remove(self.infinite_spec_constraints)
        
        # reset the array
        self.infinite_spec_constraints = []

    def AddPreConstraints(self, zTop):
        # z_vals = []
        return
        for i in range(self.history):
            # Enforce monotonically decreasing sequence constraint.
            # z = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
            # b = self.model.addMVar(1,vtype=GRB.BINARY)
            # self.model.addConstr(z == b)
            if i < self.history - 1:
                self.model.addConstr(self.Pre[i, 0] >= self.Pre[i+1, 0])
            # self.model.addConstr(self.Pre[i, 0] <= self.infinite_spec.a * self.y[:, i] - self.infinite_spec.b)
        # self.model.addConstr(self.Pre[-1, 0] <= self.infinite_spec.a * self.y[:, 0] - self.infinite_spec.b)
        # self.model.addConstr(self.Pre[-1, 0] >= self.infinite_spec.a * self.y[:, 0] - self.infinite_spec.b)
            # 

            # print(self.infinite_spec.a)
            # print(self.infinite_spec.b)
            # self.model.addConstr(self.Pre[i, 0] <= self.infinite_spec.a * self.y[:,self.horizon - i] - self.infinite_spec.b + (1-z)*self.M)
            
            # # z_vals.append(z)
            # self.model.addConstr( zTop <= z )#sum(z_vals) )
    def UpdatePreConstraints(self):
        self.model.remove(self.pre_constraints)
        self.pre_constraints = []
        for i in range(self.history):
            self.pre_constraints.append((self.alpha[i, 0] >= self.Pre[i, 0]))
        self.pre_constraints = self.model.addConstrs((_ for _ in self.pre_constraints))

    def AddFormulaConstraintsNotRecursive(self, topLevelFormula, topZ, topT, topIdx, specIdx):
        # start with a stack data structure
        stack = []
        stack.append((topLevelFormula, topZ, topT, topIdx))
        # print(topLevelFormula)
        while len(stack) > 0:
            (formula, z, t, idx) = stack.pop()
            if isinstance(formula, LinearPredicate):
                if t < 0: continue # the index is invalid due to a past time formula.
                # print(formula.a.shape)
                # print(self.y[:,t:t+1].shape)
                self.model.addConstr( formula.a.T @ self.y[:,t:t+1] - formula.b + (1-z)*self.M  >= self.rho[specIdx, 0,t] )
                b = self.model.addMVar(1,vtype=GRB.BINARY)
                self.model.addConstr(z == b)
            # elif isinstance(formula, NonlinearPredicate):
            #     raise TypeError("Mixed integer programming does not support nonlinear predicates")
            else:
                if formula.combination_type == "and":
                    for i, subformula in enumerate(formula.subformula_list):
                        t_sub = formula.timesteps[i]
                        # Check if this formula is past time.
                        if formula.pre_flag and t+t_sub >= 0:
                            if self.verbose:
                                print(f'pre_flag is true.')
                                print(formula)
                                print(subformula)
                            # time.sleep(2)
                            # print(f'pre flag is set for the parent formula of: {subformula}')
                            # continue
                            # Do something with self.pre[t+t_sub]
                            # pre_index = min([t + t_sub, self.history])
                            z_2 = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                            b_2 = self.model.addMVar(1,vtype=GRB.BINARY)

                            # if t+t_sub >= self.history:
                            # self.model.addConstr(self.infinite_spec.a * self.y[:, t+t_sub+self.history] <= self.infinite_spec.a * self.y[:, t+t_sub] + (1-z_2)*self.M)
                            # self.model.addConstr(self.infinite_spec.a * self.y[:, t+t_sub+self.history] - self.infinite_spec.b <= self.Pre[pre_index, 0] + (1-z_2)*self.M)
                            if t+t_sub < self.history:
                                # pass
                                # self.model.addConstr(self.Pre[pre_index,0] + (1-z_2) * self.M >= self.rho)
                                # self.model.addConstr(self.Pre[pre_index, 0] <= self.infinite_spec.a * self.y[:, t+t_sub] - self.infinite_spec.b)

                                # self.model.addConstr(self.Pre[pre_index, 0] + (1-z_2) * self.M >= self.infinite_spec.a * self.y[:, t+t_sub] - self.infinite_spec.b)
                                self.model.addConstr(self.Pre[t + t_sub, 0] + (1-z_2) * self.M >= self.rho[specIdx, 0, t+t_sub])
                                # self.model.addConstr(self.Pre[t + t_sub, 0] + (1-z_2) * self.M >= subformula.a*self.y[:, t+t_sub] - subformula.b)

                                # self.model.addConstr(self.Pre[pre_index,0] + (1-z_2) * self.M >= self.infinite_spec.a * self.y[:,t+t_sub] - self.infinite_spec.b)
                            else:
                                # Require that the robustness be bounded above by whatever happened history seconds ago. And history-1 seconds ago. etc.
                                # Isn't this what the main logic for adding a constraint recursively DOES???
                                # self.model.addConstr(self.rho <= self.infinite_spec.a * self.y[:, t+t_sub - self.history] - self.infinite_spec.b + (1-z_2) * self.M)
                                # self.model.addConstr(self.infinite_spec.a * self.y[:, t+t_sub] <= self.infinite_spec.a * self.y[:, t+t_sub - self.history] + (1-z_2)*self.M)
                                self.model.addConstr(subformula.a * self.y[:, t+t_sub-self.history] - subformula.b + (1-z_2)*self.M >= self.rho[specIdx, 0, t+t_sub-self.history])
                                # self.model.addConstr(subformula.a * self.y[:, t+t_sub-self.history] - subformula.b + (1-z_2)*self.M >= subformula.a*self.y[:, t+t_sub] - subformula.b)
                            
                            self.model.addConstr(z_2 == b_2)
                            self.model.addConstr( z <= z_2 )

                            # self.Pre needs to be a true Gurobi variable for this to work. Currently, it sees it as a constant.
                            # what does it equal? min(previous, LinearPredicate[t]).
                            # Pre <= a.T@y + b, at each time step.
                            # Pre >= a.T@y + b - (1-z)M, for a bunch of z's, and we require sum(z) >= 1.

                            # print(f'and: {pre_index}')
                            # self.model.addConstr(self.Pre[self.history-1, 0] >= self.infinite_spec.a * self.y[:, 0] - self.infinite_spec.b)
                        else:
                            z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                            stack.append((subformula, z_sub, t+t_sub, idx))
                            self.model.addConstr( z <= z_sub )
                else:  # combination_type == "or":
                    z_subs = []
                    for i, subformula in enumerate(formula.subformula_list):
                        # print(subformula)
                        # print(t_sub)
                        # if t+t_sub >= 0:
                        z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                        z_subs.append(z_sub)
                        t_sub = formula.timesteps[i]
                        # print(f't_sub = {t_sub}')
                        stack.append((subformula, z_sub, t+t_sub, idx)) # Negative times are handled by the predicate.

                        # print(f'historical time {t+t_sub}')

                        # Check if this formula is past time.
                        # if formula.pre_flag:
                        #     # print(f'pre flag is set for the parent formula of: {subformula}')
                        #     # continue
                        #     # Do something with self.pre[t+t_sub]
                        #     pre_index = min([t + t_sub + self.history, self.history-1])
                        #     z_2 = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                        #     self.model.addConstr(self.Pre[pre_index,0] + (1-z_2) * 2 * self.M >= self.rho)
                        #     b_2 = self.model.addMVar(1,vtype=GRB.BINARY)
                        #     self.model.addConstr(z_2 == b_2)
                        #     z_subs.append(z_2)

                        #     print(pre_index)

                    self.model.addConstr( z <= sum(z_subs) )
