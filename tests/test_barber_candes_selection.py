import itertools
import unittest

from fs_fdr import barber_candes_selection, knockoff_features_construction, utils, filter, models, forward_selection, varimp
from fs_fdr.barber_candes_selection import BarberCandesSelection


class TestBarberCandesSelection(object):
	"""docstring for TestBarberCandesSelection"""

	def setUp(self):
		
		#Simulate data
		n, p, p1 = 1000, 50, 20
		rho = 0.
		mean = 0.
		sd = 1.
		error_std = 1.
		r = (.0, 1.)

		self.x = mean + sd * np.random.normal(0., 1., size=(n, p))
		true_w = np.random.uniform(r[0], r[1], size=(p1, 1))
		negate = np.random.binomial(n=1, p=.5, size=(p1, 1))
		negate[np.where(negate==0.), :] = -1
		true_w = true_w * negate
		true_index = np.random.choice(np.arange(p), size = p1, replace=False)
		true_index = np.sort(true_index)
		xbeta = np.dot(x[:, true_index], true_w)
		pr = 1. / (1. + np.exp(-xbeta))
		self.t = (pr >= 0.5) + 0.
		self.y = xbeta + error_std * np.random.normal(0., 1., size=(n, 1))
		self.modeling = [
		{"model":"logistic regression", "params":"plain regression coef"},
		{"model":"logistic regression", "params":"ridge regression coef"},
		{"model":"logistic regression", "params":"lasso coef"},
		{"model":"logistic regression", "params":"lasso regularizer"},
		{"model":"linear regression", "params":"plain regression coef"},
		{"model":"linear regression", "params":"ridge regression coef"},
		{"model":"linear regression", "params":"lasso coef"},
		{"model":"linear regression", "params":"lasso regularizer"},
		{"model":"linear regression", "params":"forward selection coef"},
		{"model":"random forest", "params":"regression fi"},
		{"model":"random forest", "params":"classification fi"},
		{"model":"tree", "params":"regression fi"},
		{"model":"tree", "params":"classification fi"},
		{"model":"gradient boosting", "params":"regression fi"},
		{"model":"gradient boosting", "params":"classification fi"},
		{"model":"svm", "params":"regression coef"},
		{"model":"svm", "params":"classification coef"},
		{"model":"not specified", "params":"given"}
		]
		self.VI_stat = ["Diff", "Max"]
		self.split_type=[["splitting"], ["sampling", 3, 3]]
		self.prob=.5
		self.selection_method = ["knockoff", "DSS", "MSS"]

	def test_with_defaults(self):
		b = BarberCandesSelection()

	def test_modeling(self):
		pass

	def test_selection(self):
		pass




















exit()


class Knockoff(object):
	"""docstring for Knockoff"""
	def test_fixedX_knockoff_features(self):
		pass

	def test_modelX_knockoff_features(self):
		pass

	def test_optim(self):
		pass

	def test_make_pos_semi_def(self):
		pass

	def test_matrix_sqrt(self):
		pass

	def test_SDPoptim(self):
		pass

	def test_ASDPoptim(self):
		pass

	def test_augment(self):
		pass



class Filter(object):
	"""docstring for Filter"""
	def test_knockoff(self):
		pass

	def test_DSS(self):
		pass



class Model(object):
	"""docstring for Model"""
	def test_ridge_reg_coef(self):
		pass

	def test_lasso_coef(self):
		pass

	def test_lasso_regularizer(self):
		pass

	def test_fs_coef(self):
		pass

	def test_ensemble_varimp(self):
		pass

	def test_svm_coef(self):
		pass


class utils(object):
	"""docstring for utils"""
	def test_AR_corr(self):
		pass

	def test_constant_corr(self):
		pass

	def test_simulate_data(self):
		pass

	def test_FDR(self):
		pass

	def test_power(self):
		pass

	def test_FNP(self):
		pass

	def test_is_pos_semi_def(self):
		pass

	def test_bisection(self):
		pass

	def test_SigmaClusterApprox(self):
		pass

	def test_SigmaBlocksApprox(self):
		pass

	def test_SigmaEigenApprox(self):
		pass

	def test_sdp(self):
		pass

	def test_asdp(self):
		pass



if __name__ == '__main__':
    unittest.main()