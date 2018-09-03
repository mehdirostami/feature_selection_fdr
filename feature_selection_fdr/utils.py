
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from numpy.random import normal, multivariate_normal
from cvxopt import matrix, solvers
import warnings
from sklearn.cluster import AgglomerativeClustering
import multiprocessing 
import time

def plot(param_dict, layer_names, num_plots):
	"""
	layer_names is the title of weights in the state_dict() for different layers.
	num_plots is the number of histograms for each layer
	"""

	params = param_dict

	#since params is a dictionary of tensors, to get the size of each tensor saved in it, we'll use .size()
	for layer in layer_names:
		if params[layer].size(0) < num_plots:
			raise AssertionError #"Number of plots for a layer should be less than or equal to the size of that layer."
	
	
	fig, multi_plots = plt.subplots(nrows=len(layer_names), ncols=num_plots, sharex=True)

	for i in range(len(layer_names)):
		for j in range(num_plots):
			multi_plots[i, j].hist(params[layer_names[i]][j, :])


	if not os.path.exists("saved_plots"):
		os.makedirs("saved_plots")
	plt.savefig('./saved_plots/mlp_mnist.png')
	plt.show()
	

def get_batch(x, y, batch_size):
    '''
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    '''
    N = x.shape[0]
    assert N == y.shape[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :]
        batch_y = y[i:i+batch_size]
        yield (batch_x, batch_y)


def plot_learning_curve(train_loss, valid_loss):
	
	e = train_loss.shape[0]
	plt.plot(range(e), train_loss)
	plt.plot(range(e), valid_loss)
	plt.show()

	

def binary_accuracy(y, t, threshold = .5):
	"""y and t are tensors"""
	y_cat = 0 + (y >= threshold)
	#a11 = torch.sum(y_cat * t); a12 = torch.sum(y_cat * (1 - t))
	#a21 = torch.sum((1 - y_cat) * t); a22 = torch.sum((1 - y_cat) * (1 - t))
	a11 = torch.dot(y_cat, t); a12 = torch.dot(y_cat, (1 - t))
	a21 = torch.dot((1 - y_cat), t); a22 = torch.dot((1 - y_cat), (1 - t))
	print("Confusion matrix (predicted vs. observed):")
	confuse = torch.Tensor([[a11, a12], [a21, a22]])
	print(confuse)
	print("Sensitivity (%):", np.round(100*a11/(a11 + a21), 1))
	print("Specificity (%):", np.round(100*a22/(a22 + a12), 1))
	print("Accuracy (%):", np.round(100*(a11 + a22)/torch.sum(confuse), 1))

#def accuracy(y, t, threshold = .5):
#	"""y and t are tensors"""
#	y = y.data.numpy()
#	t = t.data.numpy()
#	y_cat = 0. + (y >= threshold)
#	a11 = np.dot(y_cat.T, t);
#	a12 = np.dot(y_cat.T, (1. - t))
#	a21 = np.dot((1. - y_cat).T, t)
#	a22 = np.dot((1. - y_cat).T, (1. - t))
#	print("Confusion matrix (predicted vs. observed):")
#	#confuse = np.array([[a11, a12], [a21, a22]])
#	confuse = np.vstack((np.hstack((a11, a12)), np.hstack((a21, a22))))
#	print(confuse)
#	print("Sensitivity (%):", np.round(100*a11/(a11 + a21), 1))
#	print("Specificity (%):", np.round(100*a22/(a22 + a12), 1))
#	print("Accuracy (%):", np.round(100*(a11 + a22)/np.sum(confuse), 1))
#	return("------------------------------------------")

def ROC_AUC(predprob, observed):
	fpr, tpr, _ = roc_curve(observed, predprob)
	auc = auc(fpr, tpr)
	return(fpr, tpr, auc)


###############################################################################
###############################################################################
# Simulating data
###############################################################################
###############################################################################


def AR_corr(dim, rho):

	d = np.identity(dim)
	for i in range(dim-1):
		for j in range(i+1, dim):
			d[i, j] = rho**(np.abs(i-j))
			d[j, i] = d[i, j]
	return(d)


def constant_corr(dim, rho):

	all_equal = rho*np.ones((dim, dim))
	d = all_equal - (rho-1)*np.identity(dim)
	
	return(d)



def simulate_data(n, p1, p, error_std, rho, mean, sd, r, type, corr="constant"):
	
	if p1 > p:
		raise ValueError("Number of true signals should be less than or equal to number of inputs: p >= p1")

	if rho == 0.:
		# uncorrelated inputs
		x = mean + sd * np.random.normal(0., 1., size=(n, p))
		# x = mean + sd * np.random.uniform(0., 1., size=(n, p))
	else:
		# correlated inputs 
		means = np.repeat(mean, p)
		if corr.lower() == "constant":
			cov = (sd ** 2) * constant_corr(p, rho=rho)
		elif corr.upper() == "AR(1)":
			cov = (sd ** 2) * AR_corr(p, rho = rho)
		cov_sqrt = np.linalg.cholesky(cov)
		x = means + np.dot(np.random.normal(0., 1., size = (n, p)), cov_sqrt)

	if r[0].lower() == "uniform":
		true_w = np.random.uniform(r[1], r[2], size=(p1, 1))
		negate = np.random.binomial(n=1, p=.5, size=(p1, 1))
		negate[np.where(negate==0.), :] = -1
		true_w = true_w * negate
	elif r[0].lower() == "normal":
		true_w = np.random.nromal(r[1], r[2], size=(p1, 1))
	
	true_index = np.random.choice(np.arange(p), size = p1, replace=False)
	true_index = np.sort(true_index)
	# true_index = np.arange(p1)
	xbeta = np.dot(x[:, true_index], true_w)

	if type.lower() == "regression":
		t = xbeta + error_std * np.random.normal(0., 1., size=(n, 1))
	elif type.lower() == "classification":
		pr = 1. / (1. + np.exp(-xbeta))
		t = (pr >= 0.5) + 0.

	return(x, t, true_index)



###############################################################################
###############################################################################
# FDR, power and FNP 
###############################################################################
###############################################################################


def FDR(S, TrueIndex):
	"""S is a set of integers (indexes of rejected hypotheses)
	TrueIndex is set of indeces of true signals.
	p is the number of hypotheses.
	"""
	FalseRej = [ix for ix in S if ix not in TrueIndex]
	Rej = float(S.shape[0])#number of rejections 
		
	return(np.round(1. * len(FalseRej)/max(1., Rej), 2))

def power(S, TrueIndex):
	"""S is a set of integers (indexes of rejected hypotheses)
	TrueIndex is set of indeces of true signals.
	"""
	TrueRej = [ix for ix in S if ix in TrueIndex]
	TrueIndex_size = float(TrueIndex.shape[0])
	return(np.round(1. * len(TrueRej)/max(1., TrueIndex_size), 2))

def FNP(S, TrueIndex, p):
	"""S is a set of integers (indexes of rejected hypotheses)
	TrueIndex is set of indeces of true signals.
	p is the number of hypotheses.
	""" 
	NotRej = [ix for ix in range(p) if ix not in S]
	FalseNotRej = [ix for ix in TrueIndex if ix in NotRej]
	return(np.round(1. * len(FalseNotRej) / max(1, len(NotRej)), 2))


###############################################################################
###############################################################################
# Optimization for knockoff method
###############################################################################
###############################################################################

def is_pos_semi_def(x):
		return(np.all(np.linalg.eigvals(x) >= 0.))

def bisection(Sigma, s_hat):

	gamma = 1.
	matrix = 2 * Sigma - np.diag(gamma * s_hat)

	if is_pos_semi_def(matrix):
		return(s_hat)
	
	tol = 1.;	low = 0.;	high = 1.

	while tol > .01:
		matrix = 2 * Sigma - np.diag(gamma * s_hat)
		if is_pos_semi_def(matrix):
			low = gamma
		else:
			high = gamma
		old_gamma = gamma
		gamma = (low + high)/2
		tol = np.abs(old_gamma - gamma)
	
	return(gamma * s_hat)


def SigmaClusterApprox(Sigma, block_size):
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k	
	
	num_blocks = q if r == 0 else q + 1

	cluster = AgglomerativeClustering(n_clusters=num_blocks, affinity='euclidean', linkage='ward')  
	labels = cluster.fit(Sigma).labels_
	Index = [np.argwhere(labels == j).flatten() for j in range(num_blocks)]

	subSigmas = [np.diag(Sigma[Index[j], Index[j]]) for j in range(num_blocks)]

	# for j in range(num_blocks):
	# 	_, d, Vt = np.linalg.svd(Sigma[:, Index[j]])
	# 	low_dim = np.dot(np.diag(d), Vt)
	# 	print(np.linalg.eigvals(low_dim))
	# 	subSigmas += [low_dim]

	return(subSigmas, Index)

# Sigma = np.round(np.random.uniform(size=(14, 14)), 2)
# Sigma = np.dot(Sigma.T, Sigma)
# # print(Sigma)
# block_size = 5
# # print(SigmaClusterApprox(Sigma, block_size))

# print(asdp(Sigma, block_size))

def SigmaBlocksApprox(Sigma, block_size):
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k
	
	subSigmas = [Sigma[(j*k):(j*k+k), (j*k):(j*k+k)] for j in range(q)]
	if r != 0:
		subSigmas += [Sigma[-r:, -r:]]

	return(subSigmas)


def SigmaEigenApprox(Sigma, block_size):
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k
	
	eigenvals = np.linalg.eigvals(Sigma)
	eigenvals[eigenvals < 0.] = 0.
	subSigmas = [np.diag(eigenvals[(j*k):(j*k+k)]) for j in range(q)]
	if r != 0:
		subSigmas += [np.diag(eigenvals[-r:])]
	return(subSigmas)


def sdp(Sigma):
	p = Sigma.shape[0]
	identity_p = np.identity(p)
	zero_one = np.repeat(0., p**3)
	zero_one2 = zero_one.copy()
	indexes = np.arange(p)*(1 + p + p**2)# np.arange(p)*p+ np.arange(p)*(p**2) + np.arange(p)
	zero_one[indexes] = 1.
	block_identity = zero_one.reshape(p*p, p)
	zero_one2[indexes] = -1.
	mblock_identity = zero_one2.reshape(p*p, p)
	
	c = matrix(np.repeat(-1., p))
	G = [matrix(block_identity)] + [matrix(mblock_identity)] + [matrix(block_identity)]
	h = [matrix(2.*Sigma)] + [matrix(np.zeros((p, p)))] + [matrix(identity_p)]
	solvers.options['show_progress'] = False
	sol = solvers.sdp(c, Gs=G, hs=h)['x']
	sol = np.array(sol).flatten()
	sol[sol > 1.] = 1.
	sol[sol < 0.] = 0.
	# print(os.getpid(), "\n")
	
	return(sol)


def asdp(Sigma, block_size, approx_method):
	
	approx_method = approx_method.lower()
	k = block_size
	p = Sigma.shape[0]
	r, q = p % k, p // k

	if approx_method == "selfblocks":
		subSigmas = SigmaBlocksApprox(Sigma, block_size)
	elif approx_method == "cluster":
		subSigmas, Index = SigmaClusterApprox(Sigma, block_size)

	num_blocks = q if r == 0 else q + 1
	
	#########parallel computing###########
	# pool = multiprocessing.Pool()
	# subSolutions_ = [pool.apply_async(sdp, (sub_mat, )) for sub_mat in subSigmas]
	# pool.close()
	# subSolutions = [sols.get() for sols in subSolutions_]
	subSolutions = [sdp(sub_mat) for sub_mat in subSigmas]
	
	if approx_method == "cluster":
		ordered_subSolutions = np.repeat(0., p)
		for j in range(num_blocks):
			ordered_subSolutions[Index[j]] = subSolutions[j]

		subSolutions = [ordered_subSolutions]

	s_hat = np.concatenate(subSolutions)

	s = bisection(Sigma, s_hat)
	s[s > 1.] = 1.
	s[s < 0.] = 0.			
	
	return(s)
	