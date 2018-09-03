

Feature Selection (DSS, MSS, knockoff) with False Discovery Rate Control

In the modern statistics, researchers may not fix a hypothesis, rather let the collected data speak for itself and determine what hypotheses can be tested (which features should be selected). This leads to a phenomenon called "cherry picking": in the searches through data, the inputs which describe the observed data well but cannot explain future data will likely be picked.

Controlling false discovery rate (FDR), the expected value of proportion of incorrectly selected inputs, helps pick true predictors on average.

The so called "knockoff method", introduced by Barber and Candes is a method of feature selection which controls FDR. The procedure of the knockoff method is:

Creating knockoff features
Estimating some test statistics (such as regression coefficients, decay rate etc.) using regression models (possibly with regularization) applied on the data matrix including both original and knockoff features.
Defining variable importance
Defining a threshold
Select features

The file knokoff_features_construction.py includes 3 methods of constructing knockoff features. These features are used to define variable importance for the original features. The file knockoff_selection_method.py includes codes for steps 2-5.

Constructing knockoff features seems computationally expensive. A method called "elbow" which replaces the first and second steps by application of regression models on two independent sets of data. This method proves to have the same property of controlling FDR. The file knockoff_selection_method.py includes codes for steps 1-5.

The utils.py includes functions needed to define FDR and power of selection methods.

To see examples on the knockoff and elbow methods (on linear regression models) see examples.py file. This file, additionally, includes the Benjamini-Hochberg method for feature selection.
