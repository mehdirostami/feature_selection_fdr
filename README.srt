>>> import numpy as np 
>>> from fs_fdr import barber_candes_selection, knockoff_features_construction, utils
>>> 
>>> from sklearn.ensemble import GradientBoostingClassifier
>>> #######################################################################################
>>> #### Simulate data to check the performance of the methods.
>>> #######################################################################################
>>> 
>>> 
>>> type = "classification"
>>> # First simulate some data
>>> n, p, p1 = 1000, 50, 20
>>> rho = 0.
>>> mean = 0.
>>> sd = 1.
>>> error_std = 1.
>>> r = ["uniform", .0, .5]
>>> x = mean + sd * np.random.normal(0., 1., size=(n, p))
>>> true_w = np.random.uniform(r[1], r[2], size=(p1, 1))
>>> negate = np.random.binomial(n=1, p=.5, size=(p1, 1))
>>> negate[np.where(negate==0.), :] = -1
>>> true_w = true_w * negate
>>> true_index = np.random.choice(np.arange(p), size = p1, replace=False)
>>> true_index = np.sort(true_index)
>>> xbeta = np.dot(x[:, true_index], true_w)
>>> pr = 1/(1+np.exp(-xbeta))
>>> t = (pr > .5) + 0.
>>> 
>>> q = .1
>>> # Step 2: Create knockoff features using the knockoff_method library 
>>> 
>>> # Set parameters
>>> 
>>> selection_method = "knockoff-MX"
>>> optimization = ["ASDP", "selfblocks", 50, 50]
>>> 
>>> VI_stat = "Diff"
>>> 
>>> 
>>> myknockoff = knockoff_features_construction.Knockoff(x, selection_method, optimization)
>>> knockoff_attrs = myknockoff.knockoff_features()
>>> x, x_tilda = knockoff_attrs.X, knockoff_attrs.X_tilde
>>> 
>>> 
>>> 
>>> modeling = {"model":"gradient boosting", "params":"classification fi"}
>>> 
>>> 
>>> data = [x, x_tilda, t] 
>>> knockoff_selection = barber_candes_selection.BarberCandesSelection(data, modeling, selection_method,q=q, VI_stat=VI_stat).selection()
>>> 
>>> S_knock = knockoff_selection.S
>>> FDR_UB = knockoff_selection.FDR_UB
>>> 
>>> 
>>> 
>>> fdr_knock = 100*utils.FDR(S_knock, true_index)
>>> power_knock = 100*utils.power(S_knock, true_index)
>>> fnp_knock = 100*utils.FNP(S_knock, true_index, p)
>>> print('------------Knockoff ({})-------------'.format(modeling["model"]))
>>> print("Empirical FDR: " + str(100*np.round(FDR_UB, 2)) + "%")
>>> print("FDR:  " +str(fdr_knock) + "%")
>>> print("power:  "+str(power_knock) + "%") 
>>> print("FNP:  "+str(fnp_knock) + "%")
>>> 
>>> 
>>> ##########DSS
>>> 
>>> 
>>> modeling = {"model":"gradient boosting", "params":"classification fi"}
>>> selection_method = "DSS"
>>> data = [x, t] 
>>> split_type = ["sampling", 5, 5]
>>> prob = .7
>>> DSS_selection = barber_candes_selection.BarberCandesSelection(data, modeling, selection_method,q=q).selection()
>>> 
>>> S_dss = DSS_selection.S
>>> FDR_UB = DSS_selection.FDR_UB
>>> 
>>> 
>>> 
>>> fdr_dss = 100*utils.FDR(S_dss, true_index)
>>> power_dss = 100*utils.power(S_dss, true_index)
>>> fnp_dss = 100*utils.FNP(S_dss, true_index, p)
>>> print('------------DSS ({})-------------'.format(modeling["model"]))
>>> print("Empirical FDR: " + str(100*np.round(FDR_UB, 2)) + "%")
>>> print("FDR:  " +str(fdr_dss) + "%")
>>> print("power:  "+str(power_dss) + "%") 
>>> print("FNP:  "+str(fnp_dss) + "%")
>>> 
>>> 
>>> 
>>> ############### SVM
>>> 
>>> modeling = {"model": "not specified", "params":"given"}
>>> 
>>> from sklearn.svm import SVC
>>> 
>>> svm = SVC(C=1., kernel="linear")
>>> svm_w = svm.fit(np.hstack((x, x_tilda)), t.ravel()).coef_
>>> 
>>> selection_method = "knockoff" 
>>> knockoff_selection = barber_candes_selection.BarberCandesSelection(modeling=modeling, selection_method=selection_method, w = svm_w).selection()
>>> 
>>> S_knock = knockoff_selection.S
>>> FDR_UB = knockoff_selection.FDR_UB
>>> print("empirical FDR: " + str(100*np.round(FDR_UB, 2)))
>>> 
>>> fdr_knock = 100*utils.FDR(S_knock, true_index)
>>> power_knock = 100*utils.power(S_knock, true_index)
>>> fnp_knock = 100*utils.FNP(S_knock, true_index, p)
>>> print('------------Knockoff ({})-------------'.format(modeling["model"]))
>>> print("Empirical FDR: " + str(100*np.round(FDR_UB, 2)) + "%")
>>> print("FDR:  " +str(fdr_knock) + "%")
>>> print("power:  "+str(power_knock) + "%") 
>>> print("FNP:  "+str(fnp_knock) + "%")