import numpy as np 
from scipy.interpolate import UnivariateSpline


class KnockVarImp(object):
	"""variable improtance for knockof and DSS and MSS methods"""
	def __init__(self, w, w_tilde, VI_stat):
		self.w_train = w.reshape(-1, 1)
		self.w_tilde = w_tilde.reshape(-1, 1)
		self.VI_stat = VI_stat 
		self.p = self.w_train.shape[0]

	def knockoff_variable_importance(self):
			
		if self.VI_stat.upper() == "DIFF":
			self.vi = np.abs(self.w_train) - np.abs(self.w_tilde)
		elif self.VI_stat.upper() == "MAX":
			params = np.abs(np.hstack((self.w_train, self.w_tilde)))
			self.vi = np.max(params, axis=1).reshape(-1, 1) * np.sign(np.abs(self.w_train) - np.abs(self.w_tilde)).reshape(-1, 1)
		
		return(self)


class DSSVarImp(object):
	"""docstring for DSSVarImp"""
	def __init__(self, w_train, w_valid):
		self.w_positive_train = np.abs(w_train.reshape(-1, 1))
		self.w_positive_valid = np.abs(w_valid.reshape(-1, 1))

	def empirical(self, Z_T):
		"""
		x is a numpy column vector
		"""
		sorted_Z_T = np.sort(Z_T, axis = 0).reshape(1, -1)
		I = (Z_T >= sorted_Z_T)
		F = (I.sum(1)+0.) / float(Z_T.shape[0])
		empiric = F.reshape(-1, 1)
		return(empiric)
		

	def local_exterma(self, x, f):
		
		f = f.reshape(-1, 1)
		f1 = f[:-2, :]
		f2 = f[1:-1, :]
		f3 = f[2:, :]
		cat = np.hstack((f1, np.hstack((f2, f3))))
		diff = np.hstack((f2-f1, f3-f2))

		return(x[np.where((diff[:, 0] < 0.) & (diff[:, 1] > 0.))])

	def elbow(self, Z_T, plot_label="Training"):

		w0 = np.min(Z_T)
		w1 = np.max(Z_T)
		self.empiric = self.empirical(Z_T).reshape(-1, 1)
		F0 = np.min(self.empiric)
		F1 = np.max(self.empiric)

		x = Z_T
		self.line = F0 + ((F1-F0)/(w1-w0)) * (x - w0).reshape(-1, 1)
		diff = np.abs(self.empiric - self.line)#[:self.p]
		
		self.elb = Z_T[np.argmax(diff)]

		D = Z_T
		pr, self.y = np.histogram(D, bins=1000)
		self.y = self.y[:-1] + (self.y[1] - self.y[0])/2
		self.f = UnivariateSpline(self.y, pr, s=1000)
		self.elb2 = self.local_exterma(self.y, self.f(self.y))

		return(self)


	def DSS_variable_importance(self):
		"""
		data Splitting selection
		"""
		Z_T = self.w_positive_train
		Z_V = self.w_positive_valid
		# alpha = min(.05, self.elbow(Z_T))
		elbow = self.elbow(Z_T)
		self.alpha = elbow.elb

		I_pm = (Z_V >= self.alpha) + 0
		I_pm[np.where(I_pm==0)] = -1
		I_pm = I_pm.reshape(-1, 1)
		self.vi = Z_T * I_pm

		return(self)