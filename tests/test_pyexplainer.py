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


testdata = pd.read_csv('../tests/pyexplainer_test_data/activemq-5.0.0.csv', index_col='File')
dep = testdata.columns[-4]
indep = testdata.columns[0:(len(testdata.columns) - 4)]
X_train = testdata.loc[:, indep]
y_train = testdata.loc[:, dep]
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
def test_pyexplainer_init(X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules, result):
    with pytest.raises(Exception) as e_info:
        PyExplainer(X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules)
    assert e_info.typename == result


pyExplainer = PyExplainer(X_train, y_train, indep, dep, blackbox_model)


#explain


#generate_bullet_data


#generate_html


#generate_instance_crossover_interpolation


#generate_instance_random_perturbation


#generate_risk_data


#get_risk_pred


#get_risk_score


def test_generate_progress_bar_items():
    pyExplainer.hbox_items = None
    pyExplainer.generate_progress_bar_items()
    assert isinstance(pyExplainer.hbox_items[0], widgets.Label) is True
    assert isinstance(pyExplainer.hbox_items[1], widgets.FloatProgress) is True
    assert isinstance(pyExplainer.hbox_items[2], widgets.Label) is True
    assert isinstance(pyExplainer.hbox_items[3], widgets.Label) is True


#test_generate_sliders


#on_value_change


#parse_top_rules


#retrieve_min_max_values


#run_bar_animation


@pytest.mark.parametrize('top_k_rules, result',
                         [
                             (0, 3),
                             (16, 3),
                             (1, 1),
                             (15, 15)
                         ])
def test_set_top_k_rules(top_k_rules, result):
    pyExplainer.top_k_rules = 3
    pyExplainer.set_top_k_rules(top_k_rules)
    assert pyExplainer.top_k_rules == result


#show_visualisation


@pytest.mark.parametrize('risk_score, result',
                         [
                             (53, '53%'),
                             (3.12, '3.12%')
                         ])
def test_update_risk_score(risk_score, result):
    pyExplainer.risk_data = [{'riskScore': ['8%'], 'riskPred': ['Clean']}]
    pyExplainer.update_risk_score(risk_score)
    assert pyExplainer.risk_data[0]['riskScore'][0] == result


#update_right_text


#visualise


#visualisation_data_setup
