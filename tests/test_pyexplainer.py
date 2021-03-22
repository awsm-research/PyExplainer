import pytest
import pandas as pd
import ipywidgets as widgets
from pyexplainer import __version__
from pyexplainer import pyexplainer_pyexplainer
from pyexplainer.pyexplainer_pyexplainer import PyExplainer
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier


def test_version():
    assert __version__ == '0.1.0'


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


test_data = pd.read_csv('../tests/pyexplainer_test_data/activemq-5.0.0.csv', index_col='File')
dep = test_data.columns[-4]
indep = test_data.columns[0:(len(test_data.columns) - 4)]
X_train = test_data.loc[:, indep]
y_train = test_data.loc[:, dep]
blackbox_model = RandomForestClassifier(max_depth=3, random_state=0)
blackbox_model.fit(X_train, y_train)
class_label = ['clean', 'defect']


@pytest.mark.parametrize('X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules, result',
                         [
                             (X_train, y_train, indep, dep, blackbox_model, class_label, 0, 'ValueError'),
                             (X_train, y_train, indep, dep, blackbox_model, class_label, 16, 'ValueError'),
                             (X_train, y_train, indep, dep, blackbox_model, ['clean'], 3, 'ValueError'),
                             (X_train, y_train, indep, 123, blackbox_model, class_label, 3, 'ValueError'),
                             (X_train, y_train, {}, dep, blackbox_model, class_label, 3, 'ValueError'),
                             (X_train, y_train, [], dep, blackbox_model, class_label, 3, 'ValueError'),
                             (X_train, [], indep, dep, blackbox_model, class_label, 3, 'ValueError'),
                             ([], y_train, indep, dep, blackbox_model, class_label, 3, 'ValueError')
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


py_explainer = PyExplainer(X_train, y_train, indep, dep, blackbox_model)
sample_test_data = pd.read_csv('../tests/pyexplainer_test_data/activemq-5.0.0.csv', index_col = 'File')
X_test = sample_test_data.loc[:, indep]
y_test = sample_test_data.loc[:, dep]
sample_explain_index = 0
testing_X_explain = X_test.iloc[[sample_explain_index]]
testing_y_explain = y_test.iloc[[sample_explain_index]]


# explain


# generate_bullet_data


# generate_html


# generate_instance_crossover_interpolation


# generate_instance_random_perturbation


# generate_risk_data


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


# test_generate_sliders


# on_value_change


# parse_top_rules


# retrieve_min_max_values


# run_bar_animation


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


# show_visualisation


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
                             ("testing text", 'ValueError'),
                             (123, 'ValueError')
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


# visualise


# visualisation_data_setup


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


testing_bullet_data = [{'title': '#1 Increase the values of CountStmt to more than 10',
                        'subtitle': 'Actual = 10',
                        'ticks': [2.0, 196.0],
                        'step': [1],
                        'startPoints': [0, 222.0],
                        'widths': [222.0, 228.0],
                        'colors': ['#d7191c', '#a6d96a'],
                        'markers': [10],
                        'varRef': 'CountStmt'},
                       {'title': '#2 Decrease the values of MAJOR_COMMIT to less than 1',
                        'subtitle': 'Actual = 1',
                        'ticks': [1.0, 2.0],
                        'step': [0.1],
                        'startPoints': [0, 248.0],
                        'widths': [248.0, 202.0],
                        'colors': ['#a6d96a', '#d7191c'],
                        'markers': [1],
                        'varRef': 'MAJOR_COMMIT'}]


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
                             (widgets.Label("test"), 'ValueError'),
                             ("abc", 'ValueError')
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
                               widgets.Label("test"), widgets.Label("test")], 'ValueError'),
                             ([widgets.Label("test"), testing_float_progress, widgets.Label("test")], 'ValueError')
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
                             (list(testing_X_explain), 'ValueError'),
                             (testing_y_explain, 'ValueError')
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
                             (testing_X_explain, 'ValueError'),
                             (list(testing_y_explain), 'ValueError')
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
