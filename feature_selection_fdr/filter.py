
import numpy as np 

import models


class Filter(object):
	"""
	The class implements the knockoff, DSS or MSS filters. 
	"""
	def __init__(self, data, model, params, selection_method="DSS", **kwargs):
		super(Filter, self).__init__()
		
		self.n, self.p = data[0].shape
		self.data = data
		self.params = params
		self.model = model
		self.selection_method = selection_method
		self.kwargs = kwargs		

	def knockoff(self):
		model1 = models.Model(self.model, self.params, self.selection_method, **self.kwargs)
		x, x_tilda, t = self.data[0], self.data[1], self.data[2]
		inputs = np.hstack((x, x_tilda))
		self.w = model1.fit(inputs, t).w
		
		if len(self.w.shape) == 2:
			if 1 in self.w.shape:
				self.w = self.w.flatten()
			else:
				raise ValueError("The shape of provided estimates should be (2p,), (2p, 1) or (1, 2p) (p is the original number of features.)")

		self.w_train, self.w_tilde = self.w[:self.p].reshape(-1, 1), self.w[self.p:(2 * self.p)].reshape(-1, 1)

		return(self)

	def DSS(self, split_type, prob):
		"""
		split_type is either "splitting", or "sampling" with default values for fold1=fold2=5,
		or ["sampling", fold1, fold2]
		"""
		self.split_type = split_type
		self.prob = prob

		if not isinstance(self.split_type, list):
			self.split_type = [self.split_type]

		self.split_type[0] = self.split_type[0].lower()

		x, t = self.data[0], self.data[1]
		if self.split_type[0] == "splitting":
			subsample = np.random.binomial(1, self.prob, size=self.n).reshape(-1, 1)
			subsample = np.hstack(( (subsample == 1), (subsample == 0) ))
			fold1, fold2 = 1, 1
			
		elif self.split_type[0] == "sampling":
			N = int(self.prob * self.n)
			if len(self.split_type) == 3:
				fold1, fold2 = self.split_type[1], self.split_type[2]
			elif len(self.split_type) == 1:
				fold1, fold2 = 5, 5
			else:
				raise ValueError("split_type is either a string, or a list of size 1 or size 3.")
			
			subsample = np.random.choice(np.arange(self.n), size=(N, fold1 + fold2), replace=True)

		w_train = np.zeros((self.p, 1))
		w_tilde = w_train

		for j in range(fold1):
			model1 = models.Model(self.model, self.params, **self.kwargs)

			x_train, t_train = x[subsample[:, j],:], t[subsample[:, j]]
			w_train += model1.fit(x_train, t_train).w.reshape(-1, 1)

		for j in range(fold1, fold1 + fold2):

			model2 = models.Model(self.model, self.params, **self.kwargs)

			x_tilde, t_tilde = x[subsample[:, j],:], t[subsample[:, j]]
			w_tilde += model2.fit(x_tilde, t_tilde).w.reshape(-1, 1)

		# Taking average of estimates for multiple samples or multiple splittings. When we divid dataset to half, 
		# fold1=fold2=1.
		self.w_train = w_train / fold1
		self.w_tilde = w_tilde / fold2

		self.w = np.vstack((self.w_train, self.w_tilde))

		return(self)

		