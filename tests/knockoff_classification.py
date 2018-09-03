from __future__ import print_function

import numpy as np 
from fs_fdr import knockoff_features_construction, barber_candes_selection, utils


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
pr = 1/(1+np.exp(-xbeta))
t = (pr > .5) + 0.

q = .1


modelings = [\
{"model":"logistic regression", "params":"plain regression coef"},\
{"model":"logistic regression", "params":"ridge coef"},\
{"model":"logistic regression", "params":"lasso coef"},\
{"model":"logistic regression", "params":"lasso learning rate"},\
{"model":"random forest", "params":"classification fi"},\
{"model":"tree", "params":"classification fi"},\
{"model":"gradient boosting", "params":"classification fi"},\
{"model":"svm", "params":"classification coef"}\
]

VI_stats = ["Diff"]
selection_methods = ["knockoff-MX"]
optimizations = ["SDP", "samplecorr"]

 
for selection_method in selection_methods:
	for optimization in optimizations:

		myknockoff = knockoff_features_construction.Knockoff(x, selection_method, optimization)
		knockoff_attrs = myknockoff.knockoff_features()
		x, x_tilda = knockoff_attrs.X, knockoff_attrs.X_tilde
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

f.close()

exit()
split_type = [["splitting"], ["sampling", 3, 3]]
prob = .5
selection_method = ["knockoff", "DSS", "MSS"]