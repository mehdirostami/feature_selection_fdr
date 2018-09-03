
"""
This program evaluates knockoff features of a given matrix
"""


import numpy as np 
import pandas as pd
import utils
from sklearn.preprocessing import scale
import warnings

# ["ASDP", "selfblocks", 50, 50]

class Knockoff(object):

	def __init__(self, X, selection_method, optimization, SDP_use=False, cholesky=False, **kwargs):
		"""
		Creates knockoff features only; It does not select variables.
		"""

		self.n, self.p = X.shape

		if self.n < self.p:
			raise ValueError("Number of datapoints should be at least as large as number of inputs for fixed-X knockoff. Use model-X instead.")
		
		self.kwargs = kwargs

		if not isinstance(optimization, (list, tuple)):
			optimization = [optimization.lower()]
		if len(optimization) == 1:
			self.optimization = optimization[0].lower()
			if self.optimization.lower() == 'asdp':
				self.ASDPapprox_method, self.block_size, self.min_p_ASDP = "selfblocks", 50, 50
		# To approximate covariance matrix, approximation method needs to be specified along with block size and minimum size of data for ASDP.
		elif len(optimization) == 4:
			self.optimization, self.ASDPapprox_method, self.block_size, self.min_p_ASDP = optimization[0].lower(), optimization[1].lower(), optimization[2], optimization[3]
		else:
			raise ValueError("The optimization is either a string, or a list of size either 1 or 4.\n"+\
				"			Specify the ASDPapprox_method, block_size, and min_p_ASDP.")
		

		self.selection_method = selection_method.lower()
		self.cholesky = cholesky #Method for taking square root of matrix. Choleskyesky is used if cholesky=True, and svd if False.
		self.identity_p = np.identity(self.p)
		self.SDP_use = SDP_use

		# ASDP is used only for model-X version of knockoff method.
		# if self.optimization == "asdp" and self.selection_method == "knockoff-fx":
		# 	raise AttributeError("ASDP is not available for fixed-X; Try other optimization errors.")

		self.augmented = False

		if self.selection_method == "knockoff-fx":
			if self.n < 2 * self.p: # Let's augment zero matrix to bottom of feature matrix.
				self.augmented = True
				X = np.vstack((X, np.zeros((2 * self.p - self.n, self.p))))
				self.n = 2 * self.p
				warnings.warn("Number of observations is less than twice of number of inputs (p<n<2p). Data augmentation is used!")
			
			# Scaling the inputs: First scale them to have unit norm and then numply them by square root of size of data to 
			# boost the size of parameters. 
			self.X = scale(X)
			self.U, self.d, self.Vt = np.linalg.svd(self.X, full_matrices=False)

		elif self.selection_method == "knockoff-mx":
			self.X = X
		
		self.Sigma = np.cov(self.X, rowvar=False)
		self.Sigma_inv = np.linalg.solve(self.Sigma, self.identity_p)

	def knockoff_features(self):

		if self.selection_method == "knockoff-fx":
			self.X_tilde = self.fixedX_knockoff_features()
		elif self.selection_method == "knockoff-mx":
			self.X_tilde = self.modelX_knockoff_features()

		if "t" in self.kwargs:
			if self.augmented:
				self.t = self.augment(self.t)
		
		return(self)
		
	def fixedX_knockoff_features(self):
		
		""" 
		This method/function creates fixed knockoff features (vs. randomized).
		"""
		self.min_eigenvalue = np.min(self.d) ** 2

		self.C2 , self.s = self.optim()
		# taking matrix-square root of the matrix C2, using svd of C2.
		C = self.matrix_sqrt(self.C2)

		# add zero columns to U and apply Gram-Schmit to find unitary matrix orthogonal to X (or orthogonal to u matrix in svd of X).
		U_extend = np.hstack((self.U, np.zeros((self.n, self.p))))
		Uorthog = np.linalg.qr(U_extend)[0][:, (self.p):(2 * self.p)]# The right half of the Q matrix in the QR decomposition of U_.

		out = self.X - np.dot(self.X, np.dot(self.Sigma_inv, self.s)) + np.dot(Uorthog, C)

		return(out)


	def modelX_knockoff_features(self):
		
		"""
		This method/function evaluates randomized knockoff features (mostly when inpputs are gaussians).
		"""
		self.min_eigenvalue = np.min(np.linalg.eigvals(self.Sigma))

		self.C2 , self.s = self.optim()

		scaled_X = scale(self.X, with_std=False)
		mu = self.X - np.dot(scaled_X, np.dot(self.Sigma_inv, self.s))
		C = self.matrix_sqrt(self.C2)

		standard_normal = np.random.normal(0., 1., size=(self.n, self.p))

		out = mu + np.dot(standard_normal, C)
		
		return(out)

	def optim(self):
		"""
		Three methods of optimization to find s_j: https://arxiv.org/pdf/1404.5609.pdf
		"""

		# Using semi-definite programming (SDP):
		if self.optimization == "sdp":# http://cvxopt.org/userguide/coneprog.html#s-sdpsolver
			if (self.p < 200) or (self.p >= 200 and self.SDP_use == True):
				C2, s = self.SDPoptim()
			else:
				print("WARNING: Due to computational complexity of 'SDP', the 'ASDP' optimization (blocksize = 50, approx_method='selfblocks') is used because p>= 200.")
				print("If SDP is of interest anyway, turn of the 'SDP_use' flag.")
				C2, s = self.ASDPoptim(50, approx_method="selfblocks")

		elif self.optimization == "asdp":# https://statweb.stanford.edu/~candes/papers/MF_knockoffs.pdf
			print("WARNING: When 'ASDP' optimization method is used, it's recommended to keep more correlated inputs next to each other.")
			if self.p >= self.min_p_ASDP:
				C2, s = self.ASDPoptim(self.block_size, approx_method=self.ASDPapprox_method)# selfblocks, cluster, eigen
			elif self.p < self.min_p_ASDP:
				print("WARNING: Number of features is small relative to min_p_ASDP={}; SDP is used instead of ASDP.".format(self.min_p_ASDP))
				C2, s = self.SDPoptim()
		# Equating all values of s_j to be 1-eigenvalues of Sigma. If they're larger than 1, we truncate them to 1.
		# This is a fast and relatively good method (in terms of inducing a powerful knockoff method.)
		elif self.optimization == "samplecorr":# s_j = 1.
			s = np.diag(np.abs(.5 - np.abs(np.corrcoef(self.X, rowvar=False)-self.identity_p)).mean(0))
			s[s > 1.] = 1.
			s[s < 0.] = 0.
			C2 = 2. * s - self.Sigma_inv

		# Equating all values of s_j to be twice of minimum of eigenvalues of Sigma. If it's larger than 1, we truncate it to 1.
		elif self.optimization == "min_eigenvalue":# s_j = min(1, 2*min(eigenvalue(Sigma)))
			value = min(1., 2. * self.min_eigenvalue)

			s = value * self.identity_p if value != 1. else self.identity_p
			C2 = 2. * s - (self.min_eigenvalue**2) * self.Sigma_inv
		
		C2, s = self.make_pos_semi_def(C2, s)
		
		return(C2, s)


	def make_pos_semi_def(self, C2, s):
		"""
		This can be used to correct for round off errors in calculation of SDP solutions; or can be directly used
		as an optimization method by incrementally decreasing values of s_j until C2 becomes positive semi-definite.
		"""
		eps = min(.05, 2 * self.min_eigenvalue)
		old_s = self.identity_p

		while not self.is_pos_semi_def(C2):

			s = s - eps * self.identity_p

			s_ = np.diagonal(s).flatten()
			s_[np.argwhere(s_ < eps)] = eps
			s = np.diag(s_)
			C2 = 2. * s - np.dot(s, np.dot(self.Sigma_inv, s))
			
			if np.allclose(old_s, s_):
				break
			old_s = s_
			s_min = np.min(np.diagonal(s))
			if np.allclose(s_min, 0.) or s_min < 0.:
				raise ValueError("the method has no power because C2 is not positive semi-definite.")

		if np.max(np.diagonal(s)) < .1:
			C2, s = self.power_optim(C2, s)

		return(C2, s)


	def is_pos_semi_def(self, x):
		return(np.all(np.linalg.eigvals(x) >= 0.))


	def power_optim(self, C2, s):
		"""
		To have more power, let's see if we add some epsilon to each s_j, the matrix C2 is still positive semi-definite.
		"""
		eps = .05
		s_ = s
		C2_ = C2
		
		while self.is_pos_semi_def(C2_) and np.max(np.diagonal(s_)) <= 1.:
			s = s_
			C2 = 2. * s - np.dot(s, np.dot(self.Sigma_inv, s))
			s_ = s_ + eps * self.identity_p
			C2_ = 2. * s_ - np.dot(s_, np.dot(self.Sigma_inv, s_))

		return(C2, s)

	def matrix_sqrt(self, C2):

		if self.cholesky:
			C = np.linalg.cholesky(C2)
		elif not self.cholesky:
			C2_V, C2_d, C2_Vt = np.linalg.svd(C2, full_matrices=False)
			C2_d[C2_d < 0.] = 0. 
			C = np.dot(np.diag(np.sqrt(C2_d)), C2_Vt)
		return(C)


	def SDPoptim(self):
		"""
		Defining C2 of the kncokoff paper using SDP optimization method:
		"""
		sol = utils.sdp(self.Sigma)
		s = np.diag(np.array(sol).flatten())
		C2 = 2. * s - np.dot(s, np.dot(self.Sigma_inv, s))

		return(C2, s)

	def ASDPoptim(self, block_size, approx_method):
		"""
		Defining C2 of the kncokoff paper using approximate SDP optimization method:
		https://statweb.stanford.edu/~candes/papers/MF_knockoffs.pdf
		"""
		sol = utils.asdp(self.Sigma, block_size, approx_method)
		s = np.diag(np.array(sol).flatten())
		C2 = 2. * s - np.dot(s, np.dot(self.Sigma_inv, s))
		
		return(C2, s)

	def augment(self, t):

		"""
		This function augments the response variable when we want to create fixed knockoff features anyway.
		The assumption is that 2p > n, we'd like to augment 2p-n observations to the response.
		"""
		n_before = t.shape[0]
		U = self.U[:n_before, :]
		Q = np.identity(n_before) - np.dot(U, U.T)#np.dot(self.X, np.linalg.solve(np.dot(self.X.T, self.X), self.X.T))
		sigma_hat = np.sqrt(np.dot(t.T, np.dot(Q, t)))[0, 0]
		y_hat = np.random.normal(0., sigma_hat, size=(2 * self.p - n_before, 1))
		y = np.vstack((t, y_hat))
		return(y)


#Check if knockoff-fx
# print(np.allclose(np.dot(out.T, self.X), self.Sigma-self.s))
# # print(np.isclose(np.dot(out.T, self.X), self.Sigma-self.s))
# print(np.sum(np.dot(out.T, self.X) - (self.Sigma-self.s)))

# print(np.allclose(np.dot(out.T, out), self.Sigma))
# # print(np.isclose(np.dot(out.T, out), self.Sigma))
# print(np.sum(np.dot(out.T, out) - (self.Sigma)))
# exit()
