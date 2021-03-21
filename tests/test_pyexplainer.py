import pytest
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


@pytest.mark.parametrize('X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules, result',
                         [
                             ([{'risk': '90%'}, {'ticks': [1, 10]}, {'width': [100, 200], 'description': 'abc'}],
                              "[{'risk': '90%'}, {'ticks': [1, 10]}, {'width': [100, 200], 'description': 'abc'}];"),
                             (['{}', '{}', '{}'], "[{}];")
                         ])
def test_pyexplainer_init(X_train, y_train, indep, dep, blackbox_model, class_label, top_k_rules, result):
    RandomForestClassifier

