import pytest
from pyexplainer.rulefit import FriedScale, RuleCondition, Rule, RuleEnsemble, RuleFit, Winsorizer
import numpy as np
import sys
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_virtualenv():
    return get_base_prefix_compat() != sys.prefix


INSIDE_VIRTUAL_ENV = in_virtualenv()

rule_condition_smaller = RuleCondition(1, 5, "<=", 0.4)
rule_condition_greater = RuleCondition(0, 1, ">", 0.1)

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

"""Testing RuleCondition"""


def test_rule_condition_hashing_equal1():
    assert (RuleCondition(1, 5, "<=", 0.4) == RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_equal2():
    assert (RuleCondition(1, 5, "<=", 0.5) == RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_different1():
    assert (RuleCondition(1, 4, "<=", 0.4) != RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_different2():
    assert (RuleCondition(1, 5, ">", 0.4) != RuleCondition(1, 5, "<=", 0.4))


def test_rule_condition_hashing_different3():
    assert (RuleCondition(2, 5, ">", 0.4) != RuleCondition(1, 5, ">", 0.4))


def test_rule_condition_smaller():
    np.testing.assert_array_equal(rule_condition_smaller.transform(X),
                                  np.array([1, 1, 0]))


def test_rule_condition_greater():
    np.testing.assert_array_equal(rule_condition_greater.transform(X),
                                  np.array([0, 1, 1]))


"""Testing rule"""

rule = Rule([rule_condition_smaller, rule_condition_greater], 0)


def test_rule_transform():
    np.testing.assert_array_equal(rule.transform(X),
                                  np.array([0, 1, 0]))


def test_rule_equality():
    rule2 = Rule([rule_condition_greater, rule_condition_smaller], 0)
    assert rule == rule2


def test_fried_scale():
    # FriedScale without Winsorizer
    x_scale_test = np.zeros([100, 2])
    x_scale_test[0:5, 0] = -100
    x_scale_test[5:10, 0] = 100
    x_scale_test[10:55, 0] = 1
    x_scale_test[5:55, 1] = 1
    fs = FriedScale()
    fs.train(x_scale_test)
    scaled = fs.scale(x_scale_test)
    scaled_test = x_scale_test * fs.scale_multipliers
    assert scaled.all() == scaled_test.all()
    # FriedScael with Winsorizer
    winsorizer = Winsorizer(trim_quantile=0.025)
    FriedScale(winsorizer=winsorizer)


X_short = [[0, 0], [1, 1]]
y_short = [0, 1]
clf = tree.DecisionTreeClassifier()
tree1 = clf.fit(X_short, y_short)
tree2 = clf.fit(X_short, y_short)
trees = [[tree1], [tree2]]
rule_ensemble = RuleEnsemble(tree_list=trees)


def test_rule_ensemble_two_short_tree():
    RuleEnsemble(tree_list=trees)


def test_rule_ensemble_filter_short_rules():
    rule_ensemble.filter_short_rules(3)


def test_rule_ensemble_transform():
    rule_ensemble.transform(X_short)


def test_rule_fit_tree_generator():
    X_rf, y_rf = make_classification(n_samples=1000, n_features=4, n_informative=2,
                                     n_redundant=0, random_state=0, shuffle=False)
    rf_model = RandomForestClassifier(max_depth=2, random_state=0)
    rf_model = rf_model.fit(X_rf, y_rf)
    with pytest.raises(Exception) as e_info:
        RuleFit(tree_generator=rf_model)
    assert e_info.typename == 'TypeError'


# todo
"""
RuleEnsemble
- Test filter rules with only rules that only have the "<=" operator
"""
