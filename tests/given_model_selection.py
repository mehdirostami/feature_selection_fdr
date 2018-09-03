
import numpy as np 
from fs_fdr import barber_candes_selection, knockoff_features_construction, utils

from sklearn.ensemble import GradientBoostingClassifier
#######################################################################################
#### Simulate data to check the performance of the methods.
#######################################################################################


type = "classification"
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
pr = 1/(1+np.exp(-xbeta))
t = (pr > .5) + 0.

q = .1

modeling = {"model":"not specified", "params":"given"}

model = GradientBoostingClassifier(loss="exponential")


################################################################################################################
############################DSS selection
################################################################################################################


# adaboost method

selection_method = "DSS"

subsample = np.random.binomial(1, .5, size=n)
train_index = (subsample == 1)
valid_index = (subsample == 0)



w_train = model.fit(x[train_index, :], t[train_index].ravel()).feature_importances_.reshape(-1, 1)
w_valid = model.fit(x[valid_index, :], t[valid_index].ravel()).feature_importances_.reshape(-1, 1)

w_dss = np.vstack((w_train, w_valid))

DSS_selection = barber_candes_selection.BarberCandesSelection(modeling=modeling, selection_method=selection_method, q=q, w=w_dss).selection()

S_dss = DSS_selection.S
fdr_dss = 100*utils.FDR(S_dss, true_index)
power_dss = 100*utils.power(S_dss, true_index)

print('------------{} ({}, {})-------------'.format(selection_method, modeling["model"], modeling["params"]))
print("FDR:  " +str(fdr_dss) + "%")
print("power:  "+str(power_dss) + "%")



################################################################################################################
############################ knockoff selection
################################################################################################################

optimization = "samplecorr"
selection_method = "knockoff-MX"


myknockoff = knockoff_features_construction.Knockoff(x, selection_method, optimization)
knockoff_attrs = myknockoff.knockoff_features()
x, x_tilda = knockoff_attrs.X, knockoff_attrs.X_tilde

w_knock = model.fit(np.hstack((x, x_tilda)), t.ravel()).feature_importances_

VI_stat = "Diff"

knockoff_selection = barber_candes_selection.BarberCandesSelection(modeling=modeling, selection_method=selection_method,q=q, VI_stat=VI_stat, w=w_knock).selection()

S_knock = knockoff_selection.S
fdr_knock = 100*utils.FDR(S_knock, true_index)
power_knock = 100*utils.power(S_knock, true_index)

print('------------{} ({}-{})-------------'.format(selection_method, modeling["model"], modeling["params"]))
print("FDR:  " +str(fdr_knock) + "%")
print("power:  "+str(power_knock) + "%")
