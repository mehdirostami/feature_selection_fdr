

Feature Selection with False Discovery Rate Control
---------------------------------------


Installation:

Run the following command in the command line:

pip install fsfdr


Requirements:
ubuntu 17, python 2.7


----------------------------------------


About the methods and the library:

In the modern statistics, researchers may not fix a hypothesis, rather let the collected data speak for itself and determine what hypotheses can be tested (which features should be selected). This leads to a phenomenon called "cherry picking": in the searches through data, the inputs which describe the observed data well but cannot explain future data will likely be picked.

Controlling false discovery rate (FDR), the expected value of proportion of incorrectly selected inputs, helps pick true predictors on average.

The so called "knockoff method", introduced by Barber and Candes is a method of feature selection which controls FDR. The procedure of the knockoff method is:

Creating knockoff features
Estimating some test statistics (such as regression coefficients, decay rate etc.) using regression models (possibly with regularization) applied on the data matrix including both original and knockoff features.
Defining variable importance
Defining a threshold
Select features

The file knokoff_features_construction.py includes 3 methods of constructing knockoff features. These features are used to define variable importance for the original features. The file knockoff_selection_method.py includes codes for steps 2-5.

Constructing knockoff features seems computationally expensive. A method called data splitting selection, "DSS", which replaces the first and second steps by application of regression models on two independent sets of data. This method proves to have the same property of controlling FDR. The file knockoff_selection_method.py includes codes for steps 1-5.

"MSS" (for multiple splitting selection or multiple sampling selection) is a similar method to "DSS". It needs splitting the data to multiple disjoint and randomly separated subsamples. If dataset is not large, random sampling with replacement can ba used instead as long as the sample are independent and dataset is large enough and is a representative of the original distribution so that the samples drawn are independent.

The utils.py includes functions needed to define FDR and power of selection methods. We can simulate datasets for classification and regression problems so that data follow linear and logistic models.

The library has the following flexibilities:

1) Selection methods covered:

Knockoff (model-X for normal features and fixed-X linear models), MSS, DSS

2) All of covered models and variable importance statistics are:

For classification problems:

* {"model":"logistic regression", "params":"plain regression coef"}
* {"model":"logistic regression", "params":"ridge coef"}
* {"model":"logistic regression", "params":"lasso coef"}
* {"model":"logistic regression", "params":"lasso learning rate"}
* {"model":"random forest", "params":"classification fi"}
* {"model":"tree", "params":"classification fi"}
* {"model":"gradient boosting", "params":"classification fi"}
* {"model":"svm", "params":"classification coef"}

For regression problems:

* {"model":"linear regression", "params":"plain regression coef"}
* {"model":"linear regression", "params":"ridge coef"}
* {"model":"linear regression", "params":"lasso coef"}
* {"model":"linear regression", "params":"lasso learning rate"}
* {"model":"linear regression", "params":"forward selection coef"}
* {"model":"random forest", "params":"regression fi"}
* {"model":"tree", "params":"regression fi"}
* {"model":"gradient boosting", "params":"regression fi"}
* {"model":"svm", "params":"regression coef"}


3) Optimization methods for knockoff features constructions:

* optimization = ["ASDP", "selfblocks", 50, 50] 
* optimization = "ASDP"######## with "selfblocks", 50, 50 as default for method of matrix approximation, threshold for use of ASDP and block sizes for the approximation.
* optimization = ["ASDP", "cluster", 50, 50]
* optimization = "SDP"
* optimization = "samplecorr"
* optimization = "min_eigenvalue"


4) Splitting the data for DSS and MSS can be done in a few ways:

split_type = "splitting"
prob = <a value between zero and one showing the portion of training and validation sets>

split_type = ["sampling", fold1, fold2] # folds show how many times we sample fot estimating the independent tests statistics
prob = <a value between zero and one showing the portion of the simulated subsample of the original dataset.>
