import pyexplainer
import pytest
import pandas as pd
import numpy as np
import ipywidgets as widgets
import sklearn
from sklearn.ensemble import RandomForestClassifier
from pyexplainer import __version__
from pyexplainer.pyexplainer_pyexplainer import PyExplainer
from pyexplainer import pyexplainer_pyexplainer
from sklearn.utils import check_random_state
import os
import sys
import pickle
from bs4 import BeautifulSoup


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_virtualenv():
    return get_base_prefix_compat() != sys.prefix


INSIDE_VIRTUAL_ENV = in_virtualenv()

# data paths
cwd = os.getcwd()
file_path = cwd + "/pyexplainer_test_data/activemq-5.0.0.zip"
model_file_path = cwd + '/rf_models/rf_model1.pkl'
test_file_path = cwd + "/pyexplainer_test_data/activemq-5.1.0.zip"
rule_object_path = cwd + '/rule_objects/rule_object.pyobject'
top_rules_path = cwd + '/rule_objects/top_rules.pyobject'

if INSIDE_VIRTUAL_ENV:
    cwd = os.getcwd()
    file_path = cwd + "/tests/pyexplainer_test_data/activemq-5.0.0.zip"
    model_file_path = cwd + "/tests/rf_models/rf_model1.pkl"
    test_file_path = cwd + "/tests/pyexplainer_test_data/activemq-5.1.0.zip"
    rule_object_path = cwd + "/tests/rule_objects/rule_object.pyobject"
    top_rules_path = cwd + '/tests/rule_objects/top_rules.pyobject'

train_data = pd.read_csv(file_path, index_col='File')

dep = train_data.columns[-4]
selected_features = ["ADEV", "AvgCyclomaticModified", "AvgEssential", "AvgLineBlank", "AvgLineComment",
                     "CountClassBase", "CountClassCoupled", "CountClassDerived", "CountDeclClass",
                     "CountDeclClassMethod", "CountDeclClassVariable", "CountDeclInstanceVariable",
                     "CountDeclMethodDefault", "CountDeclMethodPrivate", "CountDeclMethodProtected",
                     "CountDeclMethodPublic", "CountInput_Mean", "CountInput_Min", "CountOutput_Min", "MAJOR_LINE",
                     "MaxInheritanceTree", "MaxNesting_Min", "MINOR_COMMIT", "OWN_COMMIT", "OWN_LINE",
                     "PercentLackOfCohesion", "RatioCommentToCode"]
all_cols = train_data.columns
for col in all_cols:
    if col not in selected_features:
        all_cols = all_cols.drop(col)
indep = all_cols
X_train = train_data.loc[:, indep]
y_train = train_data.loc[:, dep]

""" write model to pickle - done 
blackbox_model = RandomForestClassifier(max_depth=3, random_state=0)
blackbox_model.fit(X_train, y_train)
with open(model_file_path, 'wb+') as file:
    pickle.dump(obj=blackbox_model, file=file)
"""

""" load model from .pkl file """
with open(model_file_path, 'rb') as file:
    blackbox_model = pickle.load(file)

class_label = ['clean', 'defect']

py_explainer = PyExplainer(X_train, y_train, indep, dep, blackbox_model)
# load data
cwd = os.getcwd()
sample_test_data = pd.read_csv(test_file_path, index_col='File')
X_test = sample_test_data.loc[:, indep]
y_test = sample_test_data.loc[:, dep]

sample_explain_index = 24
testing_X_explain = X_test.iloc[[sample_explain_index]]
testing_y_explain = y_test.iloc[[sample_explain_index]]

"""Create and Write rule_object - done
test_rule_object = py_explainer.explain(X_explain=testing_X_explain,
                                        y_explain=testing_y_explain,
                                        search_function='crossoverinterpolation',
                                        top_k=3,
                                        max_rules=30,
                                        max_iter=10000,
                                        cv=5,
                                        debug=False)
with open(rule_object_path, 'wb+') as file:
    pickle.dump(test_rule_object, file)
"""

"""Load rule_object"""
with open(rule_object_path, 'rb') as file:
    test_rule_object = pickle.load(file)

py_explainer.X_explain = testing_X_explain
py_explainer.y_explain = testing_y_explain

"""Write top_rules - done
top_rules = py_explainer.parse_top_rules(top_k_positive_rules=test_rule_object['top_k_positive_rules'],
                                         top_k_negative_rules=test_rule_object['top_k_negative_rules'])
with open(top_rules_path, 'wb+') as file:
    pickle.dump(top_rules, file)
"""

"""Load top_rules"""
with open(top_rules_path, 'rb') as file:
    top_rules = pickle.load(file)

testing_bullet_data = py_explainer.generate_bullet_data(top_rules)
testing_risk_data = py_explainer.generate_risk_data(py_explainer.X_explain)


def test_version():
    assert __version__ == '0.1.11'


@pytest.mark.parametrize('data, result',
                         [
                             (['{}', '{}', '{}'], False),
                             ([1.1231, 234.234, 123, 123, 10], False),
                             ([[], [], []], False),
                             ({"key": {}, "key2": {}}, False),
                             ([{'risk': '90%'}, {'ticks': [1, 10]}, {'width': [100, 200], 'description': 'abc'}], True)
                         ])
def test_data_validation(data, result):
    assert pyexplainer_pyexplainer.data_validation(data) is result


def test_get_default_data_and_model():
    default = pyexplainer_pyexplainer.get_default_data_and_model()
    assert isinstance(default['X_train'], pd.core.frame.DataFrame)
    assert isinstance(default['y_train'], pd.core.series.Series)
    assert isinstance(default['indep'], pd.core.indexes.base.Index)
    assert isinstance(default['dep'], str)
    assert isinstance(default['blackbox_model'], sklearn.ensemble.RandomForestClassifier)


@pytest.mark.parametrize('size, random_state, result',
                         [
                             (None, None, 15),
                             (30, None, 30),
                             (None, check_random_state(None), 15),
                             (1, check_random_state(None), 1),
                             (0, None, 15),
                             (-1, None, 15),
                             (31, None, 15),
                             ('abc', None, 15),
                             (12.23, None, 15),
                             (None, 'abc', 15),
                             (None, 12.23, 15),
                         ])
def test_id_generator(size, random_state, result):
    assert len(pyexplainer_pyexplainer.id_generator(size=size, random_state=random_state)) == result


@pytest.mark.parametrize('data, result',
                         [
                             ([{'risk': '90%'}, {'ticks': [1, 10]}, {'width': [100, 200], 'description': 'abc'}],
                              "[{'risk': '90%'}, {'ticks': [1, 10]}, {'width': [100, 200], 'description': 'abc'}];"),
                             (['{}', '{}', '{}'], "[{}];")
                         ])
def test_to_js_data(data, result):
    assert pyexplainer_pyexplainer.to_js_data(data) == result


@pytest.mark.parametrize('X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules, result',
                         [
                             (X_train, y_train, indep, dep, blackbox_model, class_label, 0, 'ValueError'),
                             (X_train, y_train, indep, dep, blackbox_model, class_label, 16, 'ValueError'),
                             (X_train, y_train, indep, dep, blackbox_model, class_label, '3', 'TypeError'),
                             (X_train, y_train, indep, dep, blackbox_model, ['clean'], 3, 'ValueError'),
                             (X_train, y_train, indep, dep, blackbox_model, 'clean', 3, 'TypeError'),
                             (X_train, y_train, indep, dep, "wrong model", class_label, 3, 'TypeError'),
                             (X_train, y_train, indep, 123, blackbox_model, class_label, 3, 'TypeError'),
                             (X_train, y_train, {}, dep, blackbox_model, class_label, 3, 'TypeError'),
                             (X_train, y_train, [], dep, blackbox_model, class_label, 3, 'TypeError'),
                             (X_train, [], indep, dep, blackbox_model, class_label, 3, 'TypeError'),
                             ([], y_train, indep, dep, blackbox_model, class_label, 3, 'TypeError')
                         ])
def test_pyexplainer_init_negative(X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules, result):
    with pytest.raises(Exception) as e_info:
        PyExplainer(X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules)
    assert e_info.typename == result


@pytest.mark.parametrize('X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules',
                         [
                             (X_train, y_train, indep, dep, blackbox_model, [1, 0], 1),
                             (X_train, y_train, indep, dep, blackbox_model, ['no_bug', 'has_bug'], 15)
                         ])
def test_pyexplainer_init_positive(X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules):
    PyExplainer(X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules)


rule_obj_keys = ['synthetic_data', 'synthetic_predictions', 'X_explain', 'y_explain', 'indep',
                 'dep', 'top_k_positive_rules', 'top_k_negative_rules', 'local_rulefit_model']


def test_auto_spearman():
    pyexplainer_pyexplainer.AutoSpearman(X_train=X_train)


@pytest.mark.parametrize('exp_X_explain, exp_y_explain, top_k, max_rules, max_iter, cv, search_function, debug, '
                         'X_train, result',
                         [
                             ([], testing_y_explain, 3, 10, 10000, 5,
                              'CrossoverInterpolation', False, X_train, 'TypeError'),
                             (testing_X_explain, testing_y_explain, 3, 10, 10000, 5,
                              'RandomPerturbation', True, X_train[:1], 'ValueError'),
                             (testing_X_explain, testing_X_explain, 3, 10, 10000, 5,
                              'CrossoverInterpolation', False, X_train, 'TypeError')
                         ])
def test_explain_negative(exp_X_explain, exp_y_explain, top_k, max_rules, max_iter, cv, search_function, debug, X_train,
                          result):
    py_explainer.X_train = X_train
    py_explainer.y_train = y_train
    with pytest.raises(Exception) as e_info:
        py_explainer.explain(exp_X_explain, exp_y_explain, top_k, max_rules, max_iter, cv, search_function, debug)
    assert e_info.typename == result


@pytest.mark.parametrize('exp_X_explain, exp_y_explain, top_k, max_rules, max_iter, cv, search_function, debug, result',
                         [
                             (testing_X_explain, testing_y_explain, 3, 10, 10000, 5,
                              'CrossoverInterpolation', False, rule_obj_keys),
                             (testing_X_explain, testing_y_explain, 3, 10, 10000, 5,
                              'RandomPerturbation', True, rule_obj_keys)
                         ])
def test_explain_positive(exp_X_explain, exp_y_explain, top_k, max_rules, max_iter, cv, search_function, debug, result):
    py_explainer.X_train = X_train
    py_explainer.y_train = y_train
    rule_object = py_explainer.explain(exp_X_explain, exp_y_explain, top_k, max_rules, max_iter, cv,
                                       search_function, debug)
    assert list(rule_object.keys()) == result
    assert isinstance(rule_object['synthetic_data'], pd.core.frame.DataFrame)
    assert isinstance(rule_object['synthetic_predictions'], np.ndarray)
    assert isinstance(rule_object['X_explain'], pd.core.frame.DataFrame)
    assert isinstance(rule_object['y_explain'], pd.core.series.Series)
    assert isinstance(rule_object['indep'], pd.core.indexes.base.Index)
    assert isinstance(rule_object['dep'], str)
    assert isinstance(rule_object['top_k_positive_rules'], pd.core.frame.DataFrame)
    assert isinstance(rule_object['top_k_negative_rules'], pd.core.frame.DataFrame)
    assert isinstance(rule_object['local_rulefit_model'], pyexplainer.rulefit.RuleFit)


@pytest.mark.parametrize('rule_object, X_explain',
                         [
                             (test_rule_object, testing_X_explain)
                         ])
def test_generate_bullet_data(rule_object, X_explain):
    py_explainer.top_k_rules = 3
    py_explainer.X_explain = X_explain
    top_rules = py_explainer.parse_top_rules(top_k_positive_rules=rule_object['top_k_positive_rules'],
                                             top_k_negative_rules=rule_object['top_k_negative_rules'])
    bullet_data = py_explainer.generate_bullet_data(top_rules)
    assert pyexplainer_pyexplainer.data_validation(bullet_data)
    for dict_data in bullet_data:
        assert isinstance(dict_data['title'], str)
        assert isinstance(dict_data['subtitle'], str)
        assert isinstance(dict_data['ticks'], list) and len(dict_data['ticks']) == 2
        assert isinstance(dict_data['step'], list) and len(dict_data['step']) == 1

        assert len(dict_data['startPoints']) == len(dict_data['widths'])
        assert isinstance(dict_data['startPoints'], list) and dict_data['startPoints'][0] == 0
        for i in range(1, len(dict_data['startPoints'])):
            dict_data['startPoints'][i] = dict_data['startPoints'][i - 1] + dict_data['widths'][i - 1]
        assert isinstance(dict_data['widths'], list) and 440 < sum(dict_data['widths']) < 460

        assert isinstance(dict_data['colors'], list)
        assert isinstance(dict_data['markers'], list) and \
               dict_data['ticks'][0] <= dict_data['markers'][0] <= dict_data['ticks'][1]
        assert isinstance(dict_data['varRef'], str)


@pytest.mark.parametrize('rule_object, result',
                         [
                             (test_rule_object, True)
                         ])
def test_generate_html(rule_object, result):
    py_explainer.visualisation_data_setup(rule_object)
    html = py_explainer.generate_html()
    assert bool(BeautifulSoup(html, "html.parser").find()) is result


@pytest.mark.parametrize('X_explain, y_explain',
                         [
                             (testing_X_explain, testing_y_explain)
                         ])
def test_generate_instance_crossover_interpolation(X_explain, y_explain):
    py_explainer.X_train = X_train
    py_explainer.y_train = y_train
    py_explainer.dep = dep
    synthetic_data = py_explainer.generate_instance_crossover_interpolation(X_explain, y_explain)
    assert isinstance(synthetic_data['synthetic_data'], pd.core.frame.DataFrame)
    assert synthetic_data['sampled_class_frequency'].equals(synthetic_data['synthetic_data'] \
                                                            .groupby([py_explainer.dep]).size())


@pytest.mark.parametrize('X_explain',
                         [
                             testing_X_explain
                         ])
def test_generate_instance_random_perturbation(X_explain):
    py_explainer.X_train = X_train
    py_explainer.y_train = y_train
    synthetic_data = py_explainer.generate_instance_random_perturbation(X_explain)
    assert isinstance(synthetic_data['synthetic_data'], pd.core.frame.DataFrame)


@pytest.mark.parametrize('X_explain, result',
                         [
                             (testing_X_explain, ['riskScore', 'riskPred'])
                         ])
def test_generate_risk_data(X_explain, result):
    risk_data = py_explainer.generate_risk_data(X_explain)
    assert list(risk_data[0].keys()) == result


@pytest.mark.parametrize('risk_data, result',
                         [
                             ([{'riskScore': ['0.5%'], 'riskPred': ['Clean']}], 'Clean'),
                             ([{'riskScore': ['89.98%'], 'riskPred': ['Defect']}], 'Defect')
                         ])
def test_get_risk_pred(risk_data, result):
    py_explainer.risk_data = risk_data
    assert py_explainer.get_risk_pred() == result


@pytest.mark.parametrize('risk_data, result',
                         [
                             ([{'riskScore': ['0.5%'], 'riskPred': ['Clean']}], 0.5),
                             ([{'riskScore': ['89%'], 'riskPred': ['Defect']}], 89)
                         ])
def test_get_risk_score(risk_data, result):
    py_explainer.risk_data = risk_data
    assert py_explainer.get_risk_score() == result


@pytest.mark.parametrize('top_k_rules, result',
                         [
                             (5, 5)
                         ])
def test_get_top_k_rules(top_k_rules, result):
    py_explainer.top_k_rules = top_k_rules
    assert py_explainer.get_top_k_rules() == result


def test_generate_progress_bar_items():
    py_explainer.hbox_items = None
    py_explainer.generate_progress_bar_items()
    assert isinstance(py_explainer.hbox_items[0], widgets.Label) is True
    assert isinstance(py_explainer.hbox_items[1], widgets.FloatProgress) is True
    assert isinstance(py_explainer.hbox_items[2], widgets.Label) is True
    assert isinstance(py_explainer.hbox_items[3], widgets.Label) is True


@pytest.mark.parametrize('bullet_data, result',
                         [
                             (testing_bullet_data, True)
                         ])
def test_generate_sliders(bullet_data, result):
    py_explainer.bullet_data = bullet_data
    sliders = py_explainer.generate_sliders()
    assert len(sliders) == len(bullet_data)
    for sld in sliders:
        assert isinstance(sld, widgets.IntSlider) or isinstance(sld, widgets.FloatSlider) is result


@pytest.mark.parametrize('top_k_positive_rules, top_k_negative_rules, top_k_rules, max_rules',
                         [
                             (
                                     test_rule_object['top_k_positive_rules'], test_rule_object['top_k_negative_rules'],
                                     1, 15),
                             (test_rule_object['top_k_positive_rules'], test_rule_object['top_k_negative_rules'], 15,
                              15),
                             (test_rule_object['top_k_positive_rules'], test_rule_object['top_k_negative_rules'], 3, 15)
                         ])
def test_parse_top_rules(top_k_positive_rules, top_k_negative_rules, top_k_rules, max_rules):
    py_explainer.top_k_rules = top_k_rules
    top_rules = py_explainer.parse_top_rules(top_k_positive_rules, top_k_negative_rules)
    assert 0 <= len(top_rules['top_tofollow_rules']) <= max_rules
    assert 0 <= len(top_rules['top_toavoid_rules']) <= max_rules


def test_retrieve_X_explain_min_max_values():
    py_explainer.X_train = X_train
    min_max = py_explainer.retrieve_X_explain_min_max_values()
    assert min_max['min_values'].equals(X_train.min())
    assert min_max['max_values'].equals(X_train.max())


@pytest.mark.parametrize('top_k_rules, result',
                         [
                             (0, 3),
                             (16, 3),
                             (1, 1),
                             (15, 15)
                         ])
def test_set_top_k_rules(top_k_rules, result):
    py_explainer.top_k_rules = 3
    py_explainer.set_top_k_rules(top_k_rules)
    assert py_explainer.top_k_rules == result


@pytest.mark.parametrize('risk_score, result',
                         [
                             (53, '53%'),
                             (3.12, '3.12%')
                         ])
def test_update_risk_score(risk_score, result):
    py_explainer.risk_data = [{'riskScore': ['8%'], 'riskPred': ['Clean']}]
    py_explainer.update_risk_score(risk_score)
    assert py_explainer.risk_data[0]['riskScore'][0] == result


@pytest.mark.parametrize('right_text, result',
                         [
                             ("testing text", 'TypeError'),
                             (123, 'TypeError')
                         ])
def test_update_right_text_negative(right_text, result):
    with pytest.raises(Exception) as e_info:
        py_explainer.hbox_items = None
        py_explainer.generate_progress_bar_items()
        py_explainer.update_right_text(right_text)
    assert e_info.typename == result


@pytest.mark.parametrize('right_text',
                         [
                             (widgets.Label("testing text"))
                         ])
def test_update_right_text_positive(right_text):
    py_explainer.hbox_items = None
    py_explainer.generate_progress_bar_items()
    py_explainer.update_right_text(right_text)


@pytest.mark.parametrize('rule_object, result',
                         [
                             (test_rule_object,
                              [test_rule_object['X_explain'], test_rule_object['y_explain'],
                               testing_bullet_data,
                               testing_risk_data])
                         ])
def test_visualisation_data_setup(rule_object, result):
    py_explainer.X_explain = None
    py_explainer.y_explain = None
    py_explainer.bullet_data = None
    py_explainer.risk_data = None
    py_explainer.visualisation_data_setup(rule_object)
    assert py_explainer.X_explain.equals(result[0])
    assert py_explainer.y_explain.equals(result[1])
    assert py_explainer.bullet_data == result[2]
    assert py_explainer.risk_data == result[3]


""" functions starting with double underscore """


@pytest.mark.parametrize('bullet_data, result',
                         [
                             ([[], []], 'ValueError'),
                             ([{}, '{}'], 'ValueError')
                         ])
def test_set_bullet_data_negative(bullet_data, result):
    py_explainer.bullet_data = None
    with pytest.raises(Exception) as e_info:
        py_explainer._PyExplainer__set_bullet_data(bullet_data)
    assert e_info.typename == result


@pytest.mark.parametrize('bullet_data, result',
                         [
                             (testing_bullet_data, testing_bullet_data)
                         ])
def test_get_and_set_bullet_data_positive(bullet_data, result):
    py_explainer.bullet_data = None
    py_explainer._PyExplainer__set_bullet_data(bullet_data)
    assert py_explainer._PyExplainer__get_bullet_data() == result


testing_float_progress = widgets.FloatProgress(value=0,
                                               min=0,
                                               max=100,
                                               bar_style='info',
                                               layout=widgets.Layout(width='40%'),
                                               orientation='horizontal')


@pytest.mark.parametrize('bullet_output, result',
                         [
                             (widgets.Label("test"), 'TypeError'),
                             ("abc", 'TypeError')
                         ])
def test_set_bullet_output_negative(bullet_output, result):
    py_explainer.bullet_output = None
    with pytest.raises(Exception) as e_info:
        py_explainer._PyExplainer__set_bullet_output(bullet_output)
    assert e_info.typename == result


testing_output = widgets.Output(layout={'border': '3px solid black'})


@pytest.mark.parametrize('bullet_output, result',
                         [
                             (testing_output, testing_output)
                         ])
def test_get_and_set_bullet_output_positive(bullet_output, result):
    py_explainer.bullet_output = None
    py_explainer._PyExplainer__set_bullet_output(bullet_output)
    assert py_explainer._PyExplainer__get_bullet_output() == result


@pytest.mark.parametrize('hbox_items, result',
                         [
                             ([testing_float_progress, widgets.Label("test"),
                               widgets.Label("test"), widgets.Label("test")], 'TypeError'),
                             ([widgets.Label("test"), testing_float_progress, widgets.Label("test")], 'TypeError')
                         ])
def test_set_hbox_items_negative(hbox_items, result):
    py_explainer.hbox_items = None
    with pytest.raises(Exception) as e_info:
        py_explainer._PyExplainer__set_hbox_items(hbox_items)
    assert e_info.typename == result


testing_hbox_items = [widgets.Label("test"), testing_float_progress, widgets.Label("test"), widgets.Label("test")]


@pytest.mark.parametrize('hbox_items, result',
                         [
                             (testing_hbox_items, testing_hbox_items)
                         ])
def test_get_and_set_hbox_items_positive(hbox_items, result):
    py_explainer.hbox_items = None
    py_explainer._PyExplainer__set_hbox_items(hbox_items)
    assert py_explainer._PyExplainer__get_hbox_items() == result


@pytest.mark.parametrize('risk_data, result',
                         [
                             ([[], []], 'ValueError'),
                             ([{}, '{}'], 'ValueError')
                         ])
def test_set_risk_data_negative(risk_data, result):
    py_explainer.risk_data = None
    with pytest.raises(Exception) as e_info:
        py_explainer._PyExplainer__set_risk_data(risk_data)
    assert e_info.typename == result


@pytest.mark.parametrize('risk_data, result',
                         [
                             ([{'riskScore': ['30.1%'], 'riskPred': ['Clean']}],
                              [{'riskScore': ['30.1%'], 'riskPred': ['Clean']}])
                         ])
def test_get_and_set_risk_data_positive(risk_data, result):
    py_explainer.risk_data = None
    py_explainer._PyExplainer__set_risk_data(risk_data)
    assert py_explainer._PyExplainer__get_risk_data() == result


@pytest.mark.parametrize('X_explain, result',
                         [
                             (list(testing_X_explain), 'TypeError'),
                             (testing_y_explain, 'TypeError')
                         ])
def test_set_X_explain_negative(X_explain, result):
    py_explainer.X_explain = None
    with pytest.raises(Exception) as e_info:
        py_explainer._PyExplainer__set_X_explain(X_explain)
    assert e_info.typename == result


@pytest.mark.parametrize('X_explain, result',
                         [
                             (testing_X_explain, testing_X_explain)
                         ])
def test_get_and_set_X_explain_positive(X_explain, result):
    py_explainer.X_explain = None
    py_explainer._PyExplainer__set_X_explain(X_explain)
    assert py_explainer._PyExplainer__get_X_explain().equals(result)


@pytest.mark.parametrize('y_explain, result',
                         [
                             (testing_X_explain, 'TypeError'),
                             (list(testing_y_explain), 'TypeError')
                         ])
def test_set_y_explain_negative(y_explain, result):
    py_explainer.y_explain = None
    with pytest.raises(Exception) as e_info:
        py_explainer._PyExplainer__set_y_explain(y_explain)
    assert e_info.typename == result


@pytest.mark.parametrize('y_explain, result',
                         [
                             (testing_y_explain, testing_y_explain)
                         ])
def test_get_set_y_explain_positive(y_explain, result):
    py_explainer.y_explain = None
    py_explainer._PyExplainer__set_y_explain(y_explain)
    assert py_explainer._PyExplainer__get_y_explain().equals(result)


"""Test the visualisation using Jupyter Notebook Kernel"""


def test_visualise():
    py_explainer.visualise(test_rule_object)


def test_visualisation_data_setup():
    py_explainer.visualisation_data_setup(test_rule_object)


def test_show_visualisation():
    py_explainer.visualisation_data_setup(test_rule_object)
    py_explainer.show_visualisation()


def test_run_bar_animation():
    py_explainer.visualisation_data_setup(test_rule_object)
    py_explainer.run_bar_animation()


def test_generate_sliders():
    py_explainer.generate_sliders()


def test_on_value_change():
    py_explainer.visualise(test_rule_object)
    change = \
        {'name': 'value',
         'old': 0.0,
         'new': 46.0,
         'owner': widgets.FloatSlider(value=46.0, continuous_update=False,
                                      description='#1 Decrease the values of PercentLackOfCohesion to less than 0',
                                      layout=widgets.Layout(height='20px', width='99%'), max=87.0,
                                      readout_format='.1f', step=1.0,
                                      style=widgets.SliderStyle(description_width='40%')),
         'type': 'change'}
    py_explainer.on_value_change(change=change, debug=True)
