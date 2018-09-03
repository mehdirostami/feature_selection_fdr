

import numpy as np 
import pandas as pd 
import statsmodels.api as sm
import statsmodels.stats.multitest as ssm 

import glmnet
from glmnet import LogitNet, ElasticNet

import forward_selection

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR, LinearSVR

from scipy.interpolate import UnivariateSpline
import warnings

class Model(object):
	"""
	This class models the augmented data (features and knockoff features)
	"""
	def __init__(self, model, params, selection_method="DSS", **kwargs):

		self.params = params.lower()
		self.model = model.lower()
		self.kwargs = kwargs 
		self.selection_method = selection_method

	def lasso_instance(self, class_name):
		if "n_lambda" in self.kwargs and "alpha" in self.kwargs:
			if self.kwargs["alpha"] != 1.:
				raise ValueError("For lasso, set 'alpha'=1.. When 'params':'lasso coef', the default is 'alpha'=1..")
			instance = class_name(**self.kwargs)
		elif "n_lambda" in self.kwargs and "alpha" not in self.kwargs:
			instance = class_name(alpha=1., **self.kwargs)
		elif "n_lambda" not in self.kwargs and "alpha" in self.kwargs:
			if self.kwargs["alpha"] != 1.:
				raise ValueError("For lasso, set 'alpha'=1.. When 'params':'lasso coef', the default is 'alpha'=1..")
			instance = class_name(n_lambda=500, **self.kwargs)
		else:
			instance = class_name(n_lambda=500, alpha=1.,**self.kwargs)
		return(instance)

	def ridge_instance(self, class_name):
		if "alpha" in self.kwargs:
			if self.kwargs["alpha"] != 0.:
				raise ValueError("For ridge, set 'alpha'=0.. When 'params':'ridge coef', the default is 'alpha'=0..")
			instance = class_name(**self.kwargs)
		else:
			instance = class_name(alpha=0.,**self.kwargs)
		return(instance)

	def logit_reg_instance(self, class_name):

		if "C" in self.kwargs:
			instance = class_name(**self.kwargs)
		else:
			instance = class_name(C=1E6, **self.kwargs)
		return(instance)

	def randforest_instance(self, class_name):

		if "n_estimators" in self.kwargs and "n_jobs" in self.kwargs:
			instance = class_name(**self.kwargs)
		elif "n_estimators" in self.kwargs and "n_jobs" not in self.kwargs:
			instance = class_name(n_jobs=2, **self.kwargs)
		elif "n_estimators" not in self.kwargs and "n_jobs" in self.kwargs:
			instance = class_name(n_estimators=100, **self.kwargs)
		else:
			instance = class_name(n_estimators=100, n_jobs=2,**self.kwargs)
		return(instance)

	def ridge_reg_coef(self, object, target, inputs):

		reg_fit = object.fit(inputs, target.ravel())
		self.w = reg_fit.coef_
		return(self)

	def lasso_coef(self, object, target, inputs):

		lasso_fit = object.fit(inputs, target.ravel())
		self.w = lasso_fit.coef_
		return(self)

	def lasso_learning_rate(self, object, target, inputs):

		lasso_fit = object.fit(inputs, target.ravel())

		coef_path = lasso_fit.coef_path_
		temp_shape = coef_path.shape
		if len(temp_shape) == 3:
			coef_path = coef_path[0,:,:]

		nlambda = coef_path.shape[1]
		index0 = (coef_path == 0.)
		index = index0.sum(1)
		index[index == nlambda] = nlambda - 1

		lambdas = lasso_fit.lambda_path_
		self.w = lambdas[index]
		return(self)


	def forwardsel_coef(self, target, inputs):#forward selection
		n, p = inputs.shape
		data = pd.DataFrame(np.hstack((target, inputs)))
		data.columns = ['t'] + ['x'+str(i+1) for i in range(p)]
		Selected_index = forward_selection.forward_selected(data, 't')
		Selected_index = [j-1 for j in Selected_index]
		num_sel = len(Selected_index)
		# p is the number of both original and knockoff features.
		w_sel = np.arange(p, p - num_sel, -1)
		self.w = np.repeat(0., p)
		self.w[Selected_index] = w_sel
		return(self)

	def ensemble_varimp(self, object, target, inputs):
		ensemble_fit = object.fit(inputs, target.ravel())
		self.w = ensemble_fit.feature_importances_
		return(self)

	def svm_coef(self, object, target, inputs):
		svm_fit = object.fit(inputs, target.ravel())
		self.w = svm_fit.coef_
		return(self)


	def fit(self, inputs, target):

		n, p = inputs.shape

		if n < p:
			warnings.warn("Number of observations {} is less than feature space size {}.".format(n, p))

		if self.model == "linear regression":

			if self.params != "lasso coef" and self.selection_method.lower() == "knockoff-fx":
				print("WARNING: It is highly recommended to use modeling{'params':'lasso coef' for 'model':'linear regression'} for fixed-X knockoffs selection method.")

			if self.params == "plain regression coef":
				# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
				instance = LinearRegression(**self.kwargs)
				self.w = self.ridge_reg_coef(instance, target, inputs).w

			elif self.params == "ridge coef":
				# https://github.com/civisanalytics/python-glmnet/blob/master/glmnet/linear.py
				instance = self.ridge_instance(ElasticNet)
				self.w = self.ridge_reg_coef(instance, target, inputs).w

			elif self.params == "lasso coef":
				# https://github.com/civisanalytics/python-glmnet/blob/master/glmnet/linear.py
				instance = self.lasso_instance(ElasticNet)
				self.w = self.lasso_coef(instance, target, inputs).w
				
			elif self.params == "lasso learning rate":
				# https://github.com/civisanalytics/python-glmnet/blob/master/glmnet/linear.py
				instance = self.lasso_instance(ElasticNet)
				self.w = self.lasso_learning_rate(instance, target, inputs).w

			elif self.params == "forward selection coef":
				# https://planspace.org/20150423-forward_selection_with_statsmodels/
				self.w = self.forwardsel_coef(target, inputs).w
				
		elif self.model == "logistic regression":

			if self.params == "plain regression coef":
				# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
				instance = self.logit_reg_instance(LogisticRegression)
				self.w = instance.fit(inputs, target.ravel()).coef_
				
			elif self.params == "ridge coef":
				# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
				instance = self.ridge_instance(LogitNet)
				self.w = self.ridge_reg_coef(instance, target, inputs).w

			elif self.params == "lasso coef":
				# https://github.com/civisanalytics/python-glmnet/blob/master/glmnet/logistic.py
				# instance = self.reg_lasso_instance()
				instance = self.lasso_instance(LogitNet)
				self.w = self.lasso_coef(instance, target, inputs).w

			elif self.params == "lasso learning rate":
				# https://github.com/civisanalytics/python-glmnet/blob/master/glmnet/logistic.py
				instance = self.lasso_instance(LogitNet)
				self.w = self.lasso_learning_rate(instance, target, inputs).w
				
			elif self.params == "forward selection coef":
				raise TypeError("In this version of the library, the forward selection is only applicable to linear models.")
					
		elif self.model == "tree":
			if self.params == "classification fi":
				# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
				instance = DecisionTreeClassifier(**self.kwargs)
			elif self.params == "regression fi":
				# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
				instance = DecisionTreeRegressor(**self.kwargs)
			self.w = self.ensemble_varimp(instance, target, inputs).w		

		elif self.model == "random forest":
			if self.params == "classification fi":
				# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
				instance = self.randforest_instance(RandomForestClassifier)
			elif self.params == "regression fi":
				# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
				instance = self.randforest_instance(RandomForestRegressor)
			self.w = self.ensemble_varimp(instance, target, inputs).w		

		elif self.model == "gradient boosting": 
			if self.params == "classification fi":
				# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
				if "loss" in self.kwargs:
					instance = GradientBoostingClassifier(**self.kwargs)
				else:
					instance = GradientBoostingClassifier(loss='exponential', **self.kwargs)
			elif self.params == "regression fi":
				# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
				if "loss" in self.kwargs:
					instance = GradientBoostingRegressor(**self.kwargs)
				else:
					instance = GradientBoostingRegressor(loss='lad', **self.kwargs)
			self.w = self.ensemble_varimp(instance, target, inputs).w		

		elif self.model == "svm":
			
			if "kernel" in self.kwargs:
				if self.params == "classification coef":
					# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
					instance = SVC(**self.kwargs)
				elif self.params == "regression coef":
					# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
					instance = LinearSVR(**self.kwargs)
			else:
				print("The linear kernel is used!")
				if self.params == "classification coef":
					# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
					instance = SVC(kernel="linear",**self.kwargs)
				elif self.params == "regression coef":
					# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
					instance = LinearSVR(**self.kwargs)#SVR(kernel="linear",**self.kwargs)
			self.w = self.svm_coef(instance, target, inputs).w

		elif self.model == "not specified": 
			pass
		else:
			raise ValueError("A valid modeling argument should be given: linear regression, logistic regression,\n"+\
							"tree, random forest, gradient boosting, svm")

		return(self)

