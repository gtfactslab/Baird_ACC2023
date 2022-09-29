# File: polytopes.py
# Purpose: matrix operations for computing controllable sets
#
# See: Predictive Control for linear and hybrid systems by F. Borrelli, A. Bemporad, M. Morari (2016)
#     for most of the math. Implementation is my own.
# 
# Fouier-Mozkin is my own implementation
#
# Note that this code file works, but could be much more efficiently programmed for small discretization steps.
#
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2022

import torch
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from scipy.optimize import LinearConstraint
from numpy import linalg

class Polytope():
	# Polytopes are H-polytopes of the form Px >= p
	def __init__(self, *args):
		if len(args) == 1:
			self.P = args[0].P
			self.p = args[0].p
		elif len(args) == 2:
			self.P = args[0] # P is a numpy array, n x m
			self.p = args[1] # p is a numpy array, n x 1
			# thus, x is m x 1. (n constraints in m dimensions)
		else:
			self.P = np.array([[np.Nan]], dtype='float32') # n = 1, by default.
			self.p = np.array([[np.Nan]], dtype='float32')
	def __str__(self):
		return f'P:\n{self.P}\np:\n{self.p}\n'

	def plot_polytope(self, m_title='Polytopic plot', m_xlabel='x_1', m_ylabel='x_2', save=False):
		# plots self.
		p_figure, p_axis = plt.subplots()
		num_equations = self.P.shape[0]
		max_x = -np.inf
		max_y = -np.inf
		min_x = np.inf
		min_y = np.inf
		for e_idx in range(num_equations):
			# Perform vertex enumeration by finding intersections with other inequalities.
			# Validate vertices by checking if H x <= h
			vertices = np.zeros((self.P.shape[1], 100))
			v_idx = 0
			for r_idx in [x for x in range(num_equations) if x != e_idx]:
				A_r = np.vstack((self.P[e_idx:e_idx+1, :], self.P[r_idx:r_idx+1, :]))
				b_r = np.vstack((self.p[e_idx:e_idx+1, :], self.p[r_idx:r_idx+1, :]))
				if np.abs(linalg.det(A_r)) > 1e-4: # Avoid singular matrix (likely parallel inequalities)
					v = linalg.inv(A_r) @ b_r
					if np.all(self.P @ v <= self.p + 1e-4): # again, add tolerance for numerical issues.
						# print('adding a vertex.')
						vertices[:, v_idx:v_idx+1] = v
						v_idx += 1

			# equation: ax + by = c is an example. Solve for y.
			# y = (1/b) (c - ax)
			#     x = -10:1:10;%-1:1:2; %-10:1:10;
			#     if H(e_idx, 2) ~= 0
			#         y = (h(e_idx) - H(e_idx, 1) * x) / H(e_idx, 2);
			#     else
			#         % x = c / a
			#         x = (h(e_idx) / H(e_idx, 1)) * ones(length(x), 1);
			#         y = -10:1:10;%-4:4:8; % -10:1:10;
			#     end
			x = vertices[0, 0:v_idx]
			y = vertices[1, 0:v_idx]
			# coords[.append(np.array((x,y)))]
			plt.plot(x, y, 'b.-')
			max_x = np.max(np.hstack((max_x, x.ravel())))
			max_y = np.max(np.hstack((max_y, y.ravel())))
			min_x = np.min(np.hstack((min_x, x.ravel())))
			min_y = np.min(np.hstack((min_y, y.ravel())))
		# p_axis.fill([0.9, 0.9, 1.1, 1.1], [0.0, 0.4, 0.0, -0.4], 'k', alpha=0.2)
		# The above line can be uncommented to generate the shaded filled in portion for dT = 0.5.
		p_axis.grid(True)
		p_axis.set_title(m_title);
		p_axis.set_xlabel(m_xlabel);
		p_axis.set_ylabel(m_ylabel);
		p_axis.set_xlim(left=min_x-0.1, right=max_x+0.1)
		p_axis.set_ylim(bottom=min_y-0.1, top=max_y+0.1)
		p_figure.subplots_adjust(bottom=0.15)
		p_figure.set_figheight(3)
		if save:
			p_figure.savefig('output/controllable_set.pdf', format='pdf')

	def to_augmented_matrix(self):
		# Converts self to an augmented matrix
		return np.hstack((self.P, self.p))

	def vstack(self, X):
		# Stacks the polytope X below self.
		self.P = np.vstack((self.P, X.P))
		self.p = np.vstack((self.p, X.p))

	def check_feasibility(self, X):
		# X is a column vector in R^n.
		return np.all(self.P @ X <= self.p)

	def minrep(self):
		# MINREP Reduces an H-polytope into its minimal representation.
		# This requires solving linear programs.
		n = self.P.shape[0]
		m = self.P.shape[1]
		entries_to_include = []
		for i in range(n):
			augmentedMatrix = self.to_augmented_matrix()
			augmentedMatrix[i,-1] += 1 # does this increase the domain of the polytope?
			# ith constraint: augmentedMatrix[i, 0:-1] x <= augmentedMatrix[i, -1]
			c = -1 * augmentedMatrix[i:i+1, 0:-1].T # min cTx s.t. a bunch of things.
			# x0 = np.zeros(m, 1)

			# TODO scipy optimize.
			# f = max_x aM(i, 1:end-1) x s.t. aM(i, 1:end-1) x < aM(i, end);
			# or: multiply by -1 and use min.
			res = linprog(c, A_ub=augmentedMatrix[:, 0:-1], b_ub=augmentedMatrix[:, -1:], bounds=(None,None))
			
			f = -res.fun#c.T @ np.expand_dims(res.x, 1) # maybe * -1? We hope that this is zero, by the way.
			# This is a linear program. Solve with scipy's built-in minimization function.
			if entries_to_include and \
			np.any(
				np.all(
					np.abs(
						augmentedMatrix[i, :] - np.hstack((self.P[entries_to_include, :], self.p[entries_to_include, :]+1))
						) < 1e-6, axis=1
					)
				):
				continue # This means that we have a duplicate, to 1e-6 precision.
			
			if (f > self.p[i] - 1e-6):
				entries_to_include.append(i)
			else:
				pass#print(f'hmm... {f}')
		# print(f'entries to include: {entries_to_include}')
		#Q = self.P[entries_to_include, :]
		#q = np.expand_dims(self.p[entries_to_include], axis=0)
		#return Polytope(Q, q)
		self.P = self.P[entries_to_include, :]
		if len(self.p.shape) < 2:
			self.p = np.expand_dims(self.p[entries_to_include], axis=0)
		else:
			self.p = self.p[entries_to_include, :]

class MatrixMath():
	def __init__(self):
		pass
	def controllable_set(A, B, dT, Hx, Hu):
		# A, B - system dynamics
		# dT - sampling rate
		# Hx, Hu - system constraints as polytopes.
		n = round(2 / dT);

		# Calculate a static matrix for filling in the '0' in the predecessor set matrix calculation
		Z = np.zeros((Hu.P.shape[0], Hx.P.shape[1]))

		for i in range(n):
			# Calculate the predecessor set.
			ctrlX_temp = np.vstack((
				np.hstack((Hx.P @ A, Hx.P @ B)),
				np.hstack((Z, Hu.P))
			))
			ctrlx_temp = np.vstack((Hx.p, Hu.p))
			ctrlX_temp_polytope = Polytope(ctrlX_temp, ctrlx_temp)
			# print('ctrlX_temp_polytope, before projection.')
			# print(ctrlX_temp_polytope)
			ctrlX_temp_polytope, _ = MatrixMath.project(ctrlX_temp_polytope, 2)
			# print('ctrlX_temp_polytope, after projection.')
			# print(ctrlX_temp_polytope)
			# print(Hx)
			# Create augmented polytope.
			Hx.vstack(ctrlX_temp_polytope)
			# And convert to a minimum representation. (Intersect sets)
			# print('Hx, before minimal representation.')
			# print(Hx)
			# Hx.plot_polytope()
			Hx.minrep()
			# print(Hx)
			# print('Hx, after minimal representation.')
			# Hx.plot_polytope()

		return Hx
	def minkowski_sum(polytope1, polytope2):
		# Create two large matrices.
		Z = np.zeros((polytope1.P.shape[0], polytope2.P.shape[1]))
		L = np.vstack((np.hstack((Z, polytope1.P)), np.hstack((polytope2.P, -polytope2.P))))
		l = np.vstack((polytope1.p, polytope2.p))
		return MatrixMath.project(Polytope(L, l), polytope1.P.shape[1])
	def linear_mapping(polytope1, A, b):
		# Applies a linear mapping to an H-polytope
		# H = P / A
		H = Polytope()
		if A == np.zeros:
			H.P = polytope1.P
			H.p = polytope1.p + b
		else:
			H.P = polytope1.P @ linalg.inv(A)
			H.h = polytope1.p + H.P @ b
		return H
	def project(polytope1, d):
		# Projects polytope1 into d dimensions
		# This function greedily implements the Fourier-Motzkin algorithm

		valid = True # default.
		debug = False # for debugging.

		n_start = polytope1.P.shape[1]

		for run_index in range(n_start - d):
			# Get the fundamental dimension of polytope1.
			n = polytope1.P.shape[0] # number of constraints
			m = polytope1.P.shape[1] # current number of dimensions

			# Project such that x[-1] = 0
			# Fourier-Motzkin: take all pairs of inequalities with opposite sign
			# coefficients of x[-1], and for each generate a new valid inequality
			# that eliminates x[-1]
			lp = polytope1.P[:, -1] > 0
			ln = polytope1.P[:, -1] < 0
			le = polytope1.P[:, -1] == 0
			list_of_positives = np.hstack((polytope1.P[lp, :], polytope1.p[lp]))
			list_of_negatives = np.hstack((polytope1.P[ln, :], polytope1.p[ln]))
			list_of_equals = np.hstack((polytope1.P[le, :], polytope1.p[le]))
			if debug:
				print(list_of_positives)
				print(list_of_negatives)
				print(list_of_equals)

			# Calculate how large inequalities needs to be
			ineq_rows = list_of_positives.shape[0] * \
						list_of_negatives.shape[0] + \
						list_of_equals.shape[0]
			inequalities = np.zeros((ineq_rows, m + 1))
			j = 0 # Initialize this index.
			for p_index in range(list_of_positives.shape[0]):
				for n_index in range(list_of_negatives.shape[0]):
					# Goal: eliminate the last value in x.
					#if np.abs(list_of_negatives[n_index, -1]) > 1e-6:
					Lambda = list_of_positives[p_index, -2] / np.abs(list_of_negatives[n_index, -2])
					inequalities[j,:] = list_of_positives[p_index, :] + Lambda * list_of_negatives[n_index, :]
					j += 1

			# step 2: propogate all inequalities that don't rely on x[-1]

			for e_index in range(list_of_equals.shape[0]):
				inequalities[j,:] = list_of_equals[e_index, :]
				j += 1

			# decode inequalities back into two matrices
			polytope1.P = inequalities[:, 0:-2]
			polytope1.p = np.expand_dims(inequalities[:, -1], axis=1)
			
			# Remove zero constraints (that is, [0 0 ... 0] <= 0)
			max_iterations = polytope1.P.shape[0] # n, but new.
			true_index = 0
			for r_index in range(max_iterations):
				if true_index > polytope1.P.shape[0]:
					break
				if np.all(polytope1.P[true_index, :] == 0):
					if polytope1.p[true_index] < 0:
						valid = False # we have an inconsistent formula
					else:
						# Remove this redundant constraint.
						t_size = polytope1.P.shape[0]
						polytope1.P = np.delete(polytope1.P, true_index, axis=0)
						polytope1.p = np.delete(polytope1.p, true_index, axis=0)
				else:
					true_index += 1 # Because this isn't C, parallelization and stuff
		return polytope1, valid
