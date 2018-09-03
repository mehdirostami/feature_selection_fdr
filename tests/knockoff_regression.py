from __future__ import print_function

import numpy as np 
from fs_fdr import knockoff_features_construction, barber_candes_selection, utils


#######################################################################################
#### Simulate data to check the performance of the methods.
#######################################################################################


n, p, p1 = 300, 30, 20
rho = 0.
mean = 0.
sd = 1.
error_std = 1.
corr = "AR(1)"
r = ["uniform", .5, 1.5]
type = "regression"
x, t, true_index = utils.simulate_data(n, p1, p, error_std, rho, mean, sd, r, type=type, corr=corr)

q = .1


from fs_fdr import barber_candes_selection
from fs_fdr import knockoff_features_construction
import utils

# First simulate some data
n, p, p1 = 1000, 50, 20
rho = 0.
mean = 0.
sd = 1.
error_std = 1.
x = mean + sd * np.random.normal(0., 1., size=(n, p))
true_w = np.random.uniform(r[1], r[2], size=(p1, 1))
negate = np.random.binomial(n=1, p=.5, size=(p1, 1))
negate[np.where(negate==0.), :] = -1
true_w = true_w * negate
true_index = np.random.choice(np.arange(p), size = p1, replace=False)
true_index = np.sort(true_index)
xbeta = np.dot(x[:, true_index], true_w)
t = xbeta + error_std * np.random.normal(0., 1., size=(n, 1))


modelings = [\
{"model":"linear regression", "params":"plain regression coef"},\
{"model":"linear regression", "params":"ridge coef"},\
{"model":"linear regression", "params":"lasso coef"},\
{"model":"linear regression", "params":"lasso learning rate"},\
{"model":"linear regression", "params":"forward selection coef"},\
{"model":"random forest", "params":"regression fi"},\
{"model":"tree", "params":"regression fi"},\
{"model":"gradient boosting", "params":"regression fi"},\
{"model":"svm", "params":"regression coef"}\
]

VI_stats = ["Diff", "Max"]
optimizations = ["SDP", "samplecorr"]


selection_methods = ["knockoff-MX"]


for selection_method in selection_methods:
	for optimization in optimizations:

		myknockoff = knockoff_features_construction.Knockoff(x, selection_method, optimization)
		knockoff_attrs = myknockoff.knockoff_features()
		x, x_tilda= knockoff_attrs.X, knockoff_attrs.X_tilde
		data = [x, x_tilda, t] 
		for modeling in modelings:
			for VI_stat in VI_stats:

				knockoff_selection = barber_candes_selection.BarberCandesSelection(data, modeling, selection_method,q=q, VI_stat=VI_stat).selection()

				S_knock = knockoff_selection.S
				fdr_knock = 100*utils.FDR(S_knock, true_index)
				power_knock = 100*utils.power(S_knock, true_index)
				with open('{}-{}.txt'.format(selection_method, type), 'a') as f:
					print('------------Knockoff ({})-------------'.format(modeling["model"]), file=f)
					print(selection_method +"-"+ optimization +"-"+ modeling["model"] +"-"+ modeling["params"] +"-"+ VI_stat, file=f)
					print("FDR:  " +str(fdr_knock) + "%", file=f)
					print("power:  "+str(power_knock) + "%", file=f)
					# print(selection_method +"-"+ optimization +"-"+ modeling["model"] +"-"+ modeling["params"] +"-"+ VI_stat)

f.close()