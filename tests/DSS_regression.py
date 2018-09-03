from __future__ import print_function

import numpy as np 
from fs_fdr import barber_candes_selection, utils
#######################################################################################
#### Simulate data to check the performance of the methods.
#######################################################################################


type = "regression"
# First simulate some data
n, p, p1 = 1000, 50, 20
rho = 0.
mean = 0.
sd = 1.
error_std = 1.
r = ["uniform", .0, .5]
x = mean + sd * np.random.normal(0., 1., size=(n, p))
true_w = np.random.uniform(r[1], r[2], size=(p1, 1))
negate = np.random.binomial(n=1, p=.5, size=(p1, 1))
negate[np.where(negate==0.), :] = -1
true_w = true_w * negate
true_index = np.random.choice(np.arange(p), size = p1, replace=False)
true_index = np.sort(true_index)
xbeta = np.dot(x[:, true_index], true_w)
t = xbeta + error_std * np.random.normal(0., 1., size=(n, 1))
q = .1

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


split_types = [["splitting"], ["sampling", 2, 2]]
probs = [.5, .7]
selection_methods = ["DSS", "MSS"]


data = [x, t]

for modeling in modelings:
	for split_type, prob, selection_method in zip(split_types, probs, selection_methods):

		DSS_selection = barber_candes_selection.BarberCandesSelection(data, modeling, selection_method, q=q, split_type=split_type, prob=prob).selection()

		S_dss = DSS_selection.S
		fdr_dss = 100*utils.FDR(S_dss, true_index)
		power_dss = 100*utils.power(S_dss, true_index)
		with open('{}-{}.txt'.format(selection_method, type), 'a') as f:
			print('------------{} ({}, {})-------------'.format(selection_method, modeling["model"], modeling["params"]), file=f)
			print(selection_method +"-"+ split_type[0] +"-"+ str(prob), file=f)
			print("FDR:  " +str(fdr_dss) + "%", file=f)
			print("power:  "+str(power_dss) + "%", file=f)

f.close()