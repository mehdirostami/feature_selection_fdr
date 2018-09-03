
import numpy as np 
import utils
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

import models
import varimp
from filter import Filter



class BarberCandesSelection(object):
	"""
	Selection procedure after parameter estimates of original features and knockoff (or estimates of features in the validation set) 
	are calculated. 
	This class has few buit-in procedures for parameter estimates. By specifying model, params and some of other key words those
	procedures can be used. It also has the capability to receive estimates evaluated separately and apply the Barber-Candes selection
	on them. If knockoff method is being used, note that only linear regression model is possible with fixed-X knockoff. Other models 
	may be used if knockoffs are constructed by model-X and features are gaussians.

	For getting the correct results, the selection method and VI_stat should be consistent with the way of the estimates are 
	calculated. For example, we can apply random forest on features and knockoff features and provide the variable importances and
	set "selection method=knockoff" and "VI_stat=ABS" for using the right feature importance statistics. If the estimates are taken
	from training and validation sets (DSS or MSS method), "selection method=DSS" or "MSS" we can use the right feature importance
	statistics that gaurantee the theoretical results.
	To check what **kwargs can be take a look at the following pages:
	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
	https://github.com/civisanalytics/python-glmnet/blob/master/glmnet/linear.py
	https://github.com/civisanalytics/python-glmnet/blob/master/glmnet/logistic.py
	https://planspace.org/20150423-forward_selection_with_statsmodels/
	http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
	http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
	http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
	http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
	http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
	http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
	http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
	http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

	In case none of these methods need to be specified and the wieghts or feature importances are already evaluated, we need to
	specify w=<vector of parameter estiamtes with length 2p> (in the place of **kwargs).
	"""

	def __init__(self, data=[], modeling={"model":"not specified", "params":"given"}, selection_method="", q=.1, offset=1., VI_stat="Diff", split_type="splitting", prob=.5, w=np.array([0]),**kwargs):
		
		params = modeling["params"].lower()
		model = modeling["model"].lower()
		self.selection_method = selection_method.lower()

		if self.selection_method == "knockoff-fx" and model != "linear regression":
			raise AttributeError("Fixed-X knockoff method method is only applicable to linear regression.")
		# If we separately want to estimate parameters/feature importances using either of DSS, MSS, knockoff and then give the 
		# estimates to te barber-candes method to select appropriate features, we should not specify any model and data,
		# set "params":"given" and w=<the estimations.>, selection_method="knockoff-mx"/"DSS". The last assignment makes sure
		# that the right feature importance will be defined according to the way that they were calculated.
		 
		if params == "given" and max(w.shape) != 1:
			self.p = max(w.shape) / 2
		elif params != "given" and max(w.shape) == 1:
			self.n, self.p = data[0].shape
		else:
			raise ValueError("When no data is given, set w=<the estimates of parameters or feature importances> and params='given'.\n" + \
				"            When data is given, an appropriate modeling should also be given and w must not be specified.")

		if model in ["decision tree", "decisiontree"]:
			model = "tree"
		if model in ["randomforest","randforest"]:
			model = "random forest"
		if model in ["gradboost", "gradientboosting"]:
			model = "gradient boosting"
		
		# The followings are the available built-in methodologies in the version of this package.
		lin_logit_mod = ["linear regression", "logistic regression"]
		ensemble_mod = ["random forest","gradient boosting", "tree"]
		svm_mod = ["svm"]
		lin_logit_par = ["plain regression coef", "ridge coef", "lasso coef", "lasso learning rate", "forward selection coef"]
		ensemble_par = ["regression fi", "classification fi"]
		svm_par = ["regression coef", "classification coef"]
		models = [lin_logit_mod, ensemble_mod, svm_mod, ["not specified"]]
		parameters = [lin_logit_par, ensemble_par, svm_par, ["given"]]
		for mod, par in zip(models, parameters):
			if ((model in mod) and (params not in par)) or ((model not in mod) and (params in par)):
				raise ValueError("The specified 'model' and 'params' in the 'modeling' dictionary are not compatible. Reconsider the modeling argument.")
		
		# if self.selection_method.lower() == "knockoff-fx" and model in ["linear regression"] and params not in ["lasso coef"]:
		# 	raise ValueError("The only 'params' available for fixed-X knockoffs is 'lasso coef'.")

		self.q = q
		self.offset = offset
		self.VI_stat = VI_stat# difference between absolute value of the estimates is the defaut value.
		self.kwargs = kwargs

		if not params[:5].lower() == "given":

			if self.selection_method[:8].lower() == "knockoff":
				knockoff_filter = Filter(data, model, params, self.selection_method, **self.kwargs).knockoff()
				self.w = knockoff_filter.w
				self.w_train, self.w_tilde = knockoff_filter.w_train, knockoff_filter.w_tilde
				
			elif self.selection_method.upper() in ["DSS", "MSS"]:

				DSS_filter = Filter(data, model, params, **self.kwargs).DSS(split_type, prob)
				self.w = DSS_filter.w
				self.w_train, self.w_tilde = DSS_filter.w_train, DSS_filter.w_tilde

			elif self.selection_method == "":
				raise ValueError("Specify a valid selection method: DSS, MSS, knockoff-MX, or knockoff-FX.")

		elif params[:5].lower() == "given":

			self.w = w
			if len(self.w.shape) == 2:
				if 1 in self.w.shape:
					self.w = self.w.flatten()
				else:
					raise ValueError("The shape of provided estimates should be (2p,), (2p, 1) or (1, 2p) (p is the original number of features.)")

			remainder = self.w.shape[0] % 2
			if remainder:
				raise ValueError("The length of vector w should be an even number.")

			self.w_train, self.w_tilde = self.w[:self.p].reshape(-1, 1), self.w[self.p:(2 * self.p)].reshape(-1, 1)

		self.w_positive_train = np.abs(self.w_train).reshape(-1, 1)
		self.w_positive_tilde = np.abs(self.w_tilde).reshape(-1, 1)
		# print(np.round(np.hstack((self.w_positive_train, self.w_positive_tilde)), 2))
		
		# print("The '{}' method is used to filter features.".format(selection_method))
		# print("The '{}' model and '{}' statistics are used for estimation.".format(model, params))
	def threshold(self): 
		
		VI = self.VI
		
		VI_positive = np.abs(VI)
		VI_positive_unique = np.unique(VI_positive[VI_positive > 0.])
		t = VI_positive_unique.reshape(1, -1)
		
		numerator = (VI <= -t).sum(0) + 0.
		denumerator = (VI >= t).sum(0) + 0.
		denumerator[denumerator == 0.] = 1. # Taking max of denumerator and 1 in the denumerator.
		
		ratio = (self.offset + numerator) / denumerator
		
		less_than_q = np.argwhere(ratio <= self.q).flatten()
		
		if not less_than_q.shape[0]:
			self.T = np.inf
			warnings.warn("The threshold is infinity; no variable is selected.")
		else:
			self.T = np.min(t[0, less_than_q])

		# Stochastic upper bound for the true FDR. 
		self.FDR_UB = ((VI <= -self.T).sum() / max(1., float((VI >= self.T).sum())))

		return(self)

	def selection(self):
		
		if self.selection_method[:8].lower() == "knockoff":
			knock_VI = varimp.KnockVarImp(self.w_train, self.w_tilde, self.VI_stat)
			self.VI = knock_VI.knockoff_variable_importance().vi.reshape(-1, 1)

		elif self.selection_method.upper() in ["DSS", "MSS"]:
			DSS_VI = varimp.DSSVarImp(self.w_train, self.w_tilde).DSS_variable_importance()
			self.VI = DSS_VI.vi
			# For visualizing DSS and MSS methods we define the following. 
			self.alpha = DSS_VI.alpha; self.empiric = DSS_VI.empiric
			self.line = DSS_VI.line; self.f = DSS_VI.f; self.y = DSS_VI.y
		# print(np.round(self.VI, 2))
		threshold = self.threshold()
		self.T = threshold.T
		self.FDR_UB = threshold.FDR_UB

		self.S = np.where(self.VI >= self.T)[0]

		return(self)

	def DSSploting(self, plot, arrow_plot, parameters):
				
		if arrow_plot:

			plt.figure(1)

			plt.subplot(211)
			plt.scatter(self.w_positive_train, self.empiric, color="b")
			plt.scatter(self.w_positive_train, self.line, color="r")
			plt.plot([self.alpha, self.alpha], [0., 1.])
			plt.ylabel("F(Z)")
			plt.ylim((0, 1.5)) 
			bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=1)
			bbox_props2 = dict(boxstyle="larrow,pad=0.3", fc="cyan", ec="b", lw=1)
			t = plt.text(0, 1.5, "Noises", ha="center", va="center", rotation=-85, size=15, bbox=bbox_props)
			t = plt.text((self.alpha+np.min(self.elb2)), 1.5, "Mix", ha="center", va="center", rotation=75, size=15, bbox=bbox_props2)
			t = plt.text(np.mean(self.elb2), 1.5, "Signals", ha="center", va="center", rotation=-50, size=15, bbox=bbox_props)

			w0 = np.min(self.w_positive_train)
			w1 = np.max(self.w_positive_train)

			plt.subplot(212)
			plt.plot(self.y, self.f(self.y))
			plt.plot([self.alpha, self.alpha], [self.f(w1), self.f(w0)], color='r')
			plt.scatter(self.alpha+.02, self.f(w0)-.02, color='r', marker="D")
			plt.annotate('elbow', xy = (self.alpha+.02, self.f(w0)-.02), xytext=(self.alpha+.02, self.f(w0)-.02))
			plt.plot([self.elb2, self.elb2], [self.f(w1), self.f(w0)], color="g")
			plt.xlabel("Z")
			plt.ylabel("f(Z)")			
			plt.show()

		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(self.w_positive_train, self.empiric, color="b")
			ax.plot(self.w_positive_train, self.line, color="r")
			ax.plot([self.alpha, self.alpha], [0., 1.])
			ax.annotate('alpha= {}'.format(self.alpha), xy = (1.5*self.alpha, .9), xytext=(1.5*self.alpha, .9))
			# plt.title("Empirical distribution of test statistics evaluated on training set")
			font = {'family': 'serif',
		        'color':  'darkred',
		        'weight': 'normal',
		        'size': 10,
		        }

			p1, rho, r, FDP, power = parameters["p1".lower()], parameters["rho".lower()], parameters["r".lower()], parameters["FDP".upper()], parameters["power".lower()]
			r1 = r[0];r2 = r[1]

			plt.text(r2/3, 0.43, 'Parameters', fontdict=font)
			plt.text(r2/3, 0.35, r'$n: ${}'.format(self.n), fontdict=font)
			plt.text(r2/3, 0.27, r'$p: ${}'.format(self.p), fontdict=font)
			plt.text(r2/3, 0.19, r'$p_1: ${}'.format(p1), fontdict=font)
			plt.text(r2/3, 0.11, r'$\rho: ${}'.format(rho), fontdict=font)
			plt.text(r2/3, 0.03, r'$r_1, r_2: ${}'.format(tuple(r)), fontdict=font)

			plt.text(r2/1.6, 0.43, 'Results', fontdict=font)
			plt.text(r2/1.6, 0.35, r'$FDP: ${}%'.format(FDP), fontdict=font)
			plt.text(r2/1.6, 0.27, r'$power: ${}%'.format(power), fontdict=font)
			plt.xlabel("Z")
			plt.ylabel("F(Z)")
			filename = "p1{}rho{}r1{}r2{}".format(p1, rho, r1, r2)
			# plt.savefig('{}.png'.format(filename))
			plt.show()
