import pandas as pd
import numpy as np
from pyexplainer.rulefit import RuleFit
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_virtualenv():
    return get_base_prefix_compat() != sys.prefix


INSIDE_VIRTUAL_ENV = in_virtualenv()

# load data
cwd = os.getcwd()
file_path = cwd + "/rulefit_test_data/boston.zip"

if INSIDE_VIRTUAL_ENV:
    cwd = os.getcwd()
    file_path = cwd + "/tests/rulefit_test_data/boston.zip"

boston_data = pd.read_csv(file_path, index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.to_numpy()

y_class = y.copy()
y_class[y_class < 21] = 0
y_class[y_class >= 21] = +1
N = X.shape[0]

# fit - classify
rf_classify = RuleFit(tree_size=4, sample_fract='default', max_rules=2000,
                      memory_par=0.01,
                      tree_generator=None,
                      rfmode='classify', lin_trim_quantile=0.025,
                      lin_standardise=True, exp_rand_tree_size=True, random_state=1)
rf_classify.fit(X, y_class, feature_names=features)

# predict
y_pred = rf_classify.predict(X)
y_proba = rf_classify.predict_proba(X)

# basic checks for probabilities
assert np.min(y_proba) >= 0
assert np.max(y_proba) <= 1

# fit - regress
X_regr, y_regr = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=1)
regr = regr.fit(X_regr, y_regr)
rf_regress = RuleFit(tree_size=4, sample_fract='default', max_rules=2000,
                     memory_par=0.01,
                     tree_generator=regr,
                     rfmode='regress', lin_trim_quantile=0.025,
                     lin_standardise=False, exp_rand_tree_size=True, random_state=1)
rf_regress.fit(X_regr, y_regr, feature_names=None)
rf_regress.predict(X_regr)
rf_regress.transform(X_regr)
# todo - test > rf_regress.get_rules(subregion=...)
rf_regress.get_rules(exclude_zero_coef=True)
rf_regress.get_feature_importance()
rf_regress.get_feature_importance(scaled=True)


def test_probabilities_match_predictions():
    # test that probabilities match actual predictions
    np.testing.assert_array_equal(np.rint(np.array(y_proba[:, 1])), y_pred)
