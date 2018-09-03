import statsmodels.formula.api as smf


def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    # formula = "{} ~ {} + 1".format(response,' + '.join(selected))
    # model = smf.ols(formula, data).fit()
    # return model
    S = [int(s.strip('x')) for s in selected]
    return(S)


# import pandas as pd
# import numpy as np 
# import utils

# n, p1, p = 1000, 5, 20
# rho = 0.0
# mean = 0.
# sd = 1.
# error_std = 1.
# corr = "AR(1)"
# r = ["uniform", 0., 3.5]
# type="regression"

# x, t, true_index = utils.simulate_data(n, p1, p, error_std, rho, mean, sd, r, type, corr="AR(1)")

# data = pd.DataFrame(np.hstack((t, x)))

# data.columns = ['t'] + ['x'+str(i+1) for i in range(p)]

# S = forward_selected(data, 't')



# model = forward_selected(data, 't')
# print model.summary()
# print model.model.formula
# print model.rsquared_adj
# print model.params
# print(true_index+1)
# print model.tvalues
# print model.pvalues