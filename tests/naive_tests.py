import csv
import numpy as np 
import pandas as pd
import utils
import barber_candes_selection
import knockoff_features_construction
import matplotlib.pyplot as plt
import timeit


#######################################################################################
#### Simulate data to check the performance of the methods.
#######################################################################################

# Step 1: Create data using the utils library


#Set parameters
n, p, p1 = 100, 50, 20
rho = 0.
mean = 0.
sd = 1.
error_std = 1.
corr = "AR(1)"
r = ["uniform", .5, 1.5]

x, t, true_index = utils.simulate_data(n, p1, p, error_std, rho, mean, sd, r, type="classification", corr=corr)

q = .1


# modelings = 
# [
# {"model":"logistic regression", "params":"plain regression coef"},
# {"model":"logistic regression", "params":"ridge regression coef"},
# {"model":"logistic regression", "params":"lasso coef"},
# {"model":"logistic regression", "params":"lasso regularizer"},
# {"model":"linear regression", "params":"plain regression coef"},
# {"model":"linear regression", "params":"ridge regression coef"},
# {"model":"linear regression", "params":"lasso coef"},
# {"model":"linear regression", "params":"lasso regularizer"},
# {"model":"linear regression", "params":"forward selection coef"},
# {"model":"random forest", "params":"regression fi"},
# {"model":"random forest", "params":"classification fi"},
# {"model":"tree", "params":"regression fi"},
# {"model":"tree", "params":"classification fi"},
# {"model":"gradient boosting", "params":"regression fi"},
# {"model":"gradient boosting", "params":"classification fi"},
# {"model":"svm", "params":"regression coef"},
# {"model":"svm", "params":"classification coef"},
# {"model":"not specified", "params":"given"}
# ]

# VI_stats = ["Diff", "Max"]
# knockoff_creations = ["MX", "FX"]
# optimizations = ["SDP", "ASDP", "samplecorr"]

modelings = [\
{"model":"logistic regression", "params":"plain regression coef"}\
]
VI_stats = ["Diff"]
knockoff_creations = ["MX"]
optimizations = ["SDP"]


for knockoff_creation in knockoff_creations:
	for optimization in optimizations:

		myknockoff = knockoff_features_construction.Knockoff(x, t, knockoff_creation, optimization, print_s)
		knockoff_attrs = myknockoff.knockoff_features()
		x, x_tilda, t = knockoff_attrs.X, knockoff_attrs.X_tilde, knockoff_attrs.t

		for modeling in modelings:
			for VI_stat in VI_stats:


				data = [x, x_tilda, t] 
				selection_method = "knockoff"
				knockoff_selection = barber_candes_selection.BarberCandesSelection(data, modeling, selection_method,q=q, VI_stat=VI_stat).selection()

				S_knock = knockoff_selection.S
				fdr_knock = 100*utils.FDR(S_knock, true_index)
				power_knock = 100*utils.power(S_knock, true_index)
				print('------------Knockoff ({})-------------'.format(modeling["model"]))
				print("FDR:  " +str(fdr_knock) + "%")
				print("power:  "+str(power_knock) + "%") 

exit()
split_type = [["splitting"], ["sampling", 3, 3]]
prob = .5
selection_method = ["knockoff", "DSS", "MSS"]