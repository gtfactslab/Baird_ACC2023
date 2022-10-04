# File: rta-stlpy.py
# Purpose: Implements the algorithm in "Guaranteeing Signal Temporal Logic Safety Specifications with Runtime Assurance"
#
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2022

import math
import numpy as np
import stlpy
import time
from stlpy.systems import LinearSystem
from stlpy.STL import LinearPredicate
from stlpy.solvers import GurobiMICPSolver
from matplotlib import pyplot as plt # Python plotting
from matplotlib import rc
from polytopes import Polytope, MatrixMath

rc('text', usetex=True)

dT = 1 # for first run
# dT = 0.25 # uncomment for other plots

x0 = np.array([[1.1], [0.]]) # uncomment for constant input first run in the paper
# x0 = np.array([[0.0], [0.]]) # uncomment for constant input second run
# x0 = np.array([[0.9], [0.]]) # uncomment for sinusoidal input

sim_length_seconds = 30 # in seconds
t_np = np.arange(0, sim_length_seconds, dT)
sim_length = t_np.shape[0]

u_nominal = np.ones((1, sim_length)) # uncomment for constant input, first & second runs.
# u_nominal = 0.1 * np.sin(t_np * 3.1415926535 / 8).reshape((1, sim_length)) # uncomment for sinusoidal input, third run.
print('nominal input vector')
print(u_nominal)

u0 = np.array([[0.0]])
u_max = 1

# Define the system. Double integrator.
A = np.array([[1., dT],
			  [0, 1.]])
B = np.array([[0.],
			  [dT]])
C = np.array([[1, 0.]]) # output is position.
D = np.array([[0.]])

sys = LinearSystem(A, B, C, D) # Create the linear system.

# Build the STL Formula. "Positive Normal Form" (can only negate predicates.)
lb = LinearPredicate(1, 0.9)# upper bound, A.T x - b >= 0
ub = LinearPredicate(-1, -1.1) # lower bound, A.T x - b >= 0
x_in_bounds = lb & ub

# Recall: implication a=>b is ~a | b.
pi = x_in_bounds | x_in_bounds.always(0, round(2/dT)-1).eventually(0, round(2 / dT))
print(pi)

horizon = 2 * round(2 / dT)
fts = horizon # future time steps to project the system out. (N=horizon, b=1, applying Proposition 1 in the paper).

# Create data structures to save past states and inputs.
x_hist = np.zeros((2, sim_length))
x_hist[:, 0:1] = x0
u_hist = np.zeros((1, sim_length))

print('System dimensions.')
print(f'p: {sys.p}')
print(f'n: {sys.n}')
print(f'm: {sys.m}')

# Create two polytopes: one representing bounds on x, the other representing bounds on u.
H_x = np.array([[1,0], [-1, 0], [0, 1], [0, -1]])
h_x = np.array([[1.1], [-0.9], [1], [1]]) # velocity bounds are an initial overestimate.
H_u = np.array([[1], [-1]])
h_u = np.array([[u_max], [u_max]])
Hx = Polytope(H_x, h_x)
Hu = Polytope(H_u, h_u)
Qp = MatrixMath.controllable_set(A, B, dT, Hx, Hu)
Qp.plot_polytope(m_title=f'Maximal control invariant set, $\Delta t={dT}$', m_xlabel='$x_1$', m_ylabel='$x_2$', save=False)

basic_line = np.ones((sim_length)) # for plotting "y in bounds"

start_time = time.time() # to measure execution time.
for i in range(1, sim_length):
	solver = GurobiMICPSolver(spec=pi, sys=sys, x0=x_hist[:, 0:i], T=i+fts, robustness_cost=False, horizon=horizon)
	solver.AddControlBounds(-u_max, u_max)
	solver.AddRecursiveFeasibilityConstraint(Qp)
	solver.AddLPCost()
	solver.AddLPConstraints(i-1, u_nominal[0, i-1:i])
	solution = solver.Solve()
	print(f'i = {i}')
	print(solution)

	x_1 = solution[0]
	u_1 = solution[1]
	obj_1 = solution[4]

	x_hist[:, i:i+1] = x_1[:, i:i+1]
	u_hist[0, i-1] = u_1[0, i-1]

print("--- Execution time: %s seconds ---" % (time.time() - start_time))

# # uncomment to print the results as individual plots.
# position (x_1)
# x_figure, x_axis = plt.subplots()
# x_axis.plot(t_np, x_hist[0, 0:sim_length], 'b.-')
# x_axis.plot(t_np, basic_line * 1.1, 'r')
# x_axis.plot(t_np, basic_line * 0.9, 'r')
# x_axis.grid(True)
# x_axis.set_title('Output $y[t]$')
# x_axis.set_xlabel('$t$ (s)')
# x_axis.set_ylabel('$y[t]$')
# x_figure.set_figheight(4)
# x_figure.savefig('output/x.pdf', format='pdf')

# # velocity (x_2)
# v_figure, v_axis = plt.subplots()
# v_axis.plot(t_np, x_hist[1, 0:sim_length], 'b.-')
# v_axis.grid(True)
# v_axis.set_title('$x_2[t]$')
# v_axis.set_xlabel('t (s)')
# v_axis.set_ylabel('$x_2[t]$')
# v_figure.set_figheight(4)
# v_figure.savefig('output/v.pdf', format='pdf')

# # input
# u_figure, u_axis = plt.subplots()
# i_u_1, = u_axis.plot(t_np, u_hist[0, 0:sim_length], 'b.-', alpha=0.6)
# i_u_2, = u_axis.plot(t_np, u_nominal[0, 0:sim_length], 'r-', alpha=0.6)
# u_axis.grid(True)
# u_axis.set_title('Input $u[t]$')
# u_axis.set_xlabel('$t$ (s)')
# u_axis.set_ylabel('$u[t]$')
# u_axis.legend([i_u_1, i_u_2], ['Filtered', 'Nominal'], loc="lower right")
# u_figure.set_figheight(4)
# u_figure.savefig('output/u.pdf', format='pdf')
# plt.show()

main_figure, (x_axis, u_axis) = plt.subplots(2, 1)

i_x_1, = x_axis.plot(t_np, x_hist[0, 0:sim_length], 'b.-')
i_x_2, = x_axis.plot(t_np, basic_line * 1.1, 'r')
x_axis.plot(t_np, basic_line * 0.9, 'r')
x_axis.grid(True)
x_axis.set_title('Output $y[t]$')
x_axis.set_xlabel('$t$ (s)')
x_axis.set_ylabel('$y[t]$')
x_axis.legend([i_x_1, i_x_2], ['A safe trajectory', '$y$ in bounds'], loc="lower right")

i_u_1, = u_axis.plot(t_np, u_hist[0, 0:sim_length], 'b.-', alpha=0.7)
i_u_2, = u_axis.plot(t_np, u_nominal[0, 0:sim_length], 'k-', alpha=0.8)
u_axis.grid(True)
u_axis.set_title('Input $u[t]$')
u_axis.set_xlabel('$t$ (s)')
u_axis.set_ylabel('$u[t]$')
u_axis.legend([i_u_1, i_u_2], ['Filtered', 'Nominal'], loc="lower right")

main_figure.subplots_adjust(hspace=0.5)
main_figure.set_figheight(6)

main_figure.savefig('output/u_and_x.pdf', format='pdf')

plt.show()