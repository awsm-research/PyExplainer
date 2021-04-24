import copy
import math
import os
import random
import re
import sys
import string
import warnings
import ipywidgets as widgets
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from IPython.core.display import display, HTML
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, all_estimators
from .rulefit import RuleFit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pickle


def AutoSpearman(X_train, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5):
    """An automated feature selection approach that address collinearity and multicollinearity.
    For more information, please kindly refer to the `paper <https://ieeexplore.ieee.org/document/8530020>`_.

    Parameters
    ----------
    X_train : :obj:`pd.core.frame.DataFrame`
        The X_train data to be processed
    correlation_threshold : :obj:`float`
        Threshold value of correalation.
    correlation_method : :obj:`str`
        Method for solving the correlation between the features.
    VIF_threshold : :obj:`int`
        Threshold value of VIF score.
    """
    X_AS_train = X_train.copy()
    AS_metrics = X_AS_train.columns
    count = 1

    # (Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test.
    print('(Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test')
    while True:
        corrmat = X_AS_train.corr(method=correlation_method)
        top_corr_features = corrmat.index
        abs_corrmat = abs(corrmat)

        # identify correlated metrics with the correlation threshold of the threshold
        highly_correlated_metrics = ((corrmat > correlation_threshold) | (corrmat < -correlation_threshold)) & (
                corrmat != 1)
        n_correlated_metrics = np.sum(np.sum(highly_correlated_metrics))
        if n_correlated_metrics > 0:
            # find the strongest pair-wise correlation
            find_top_corr = pd.melt(abs_corrmat, ignore_index=False)
            find_top_corr.reset_index(inplace=True)
            find_top_corr = find_top_corr[find_top_corr['value'] != 1]
            top_corr_index = find_top_corr['value'].idxmax()
            top_corr_i = find_top_corr.loc[top_corr_index, :]

            # get the 2 correlated metrics with the strongest correlation
            correlated_metric_1 = top_corr_i[0]
            correlated_metric_2 = top_corr_i[1]
            print('> Step', count, 'comparing between', correlated_metric_1, 'and', correlated_metric_2)

            # compute their correlation with other metrics outside of the pair
            correlation_with_other_metrics_1 = np.mean(abs_corrmat[correlated_metric_1][
                                                           [i for i in top_corr_features if
                                                            i not in [correlated_metric_1, correlated_metric_2]]])
            correlation_with_other_metrics_2 = np.mean(abs_corrmat[correlated_metric_2][
                                                           [i for i in top_corr_features if
                                                            i not in [correlated_metric_1, correlated_metric_2]]])
            print('>>', correlated_metric_1, 'has the average correlation of',
                  np.round(correlation_with_other_metrics_1, 3), 'with other metrics')
            print('>>', correlated_metric_2, 'has the average correlation of',
                  np.round(correlation_with_other_metrics_2, 3), 'with other metrics')
            # select the metric that shares the least correlation outside of the pair and exclude the other
            if correlation_with_other_metrics_1 < correlation_with_other_metrics_2:
                exclude_metric = correlated_metric_2
            else:
                exclude_metric = correlated_metric_1
            print('>>', 'Exclude', exclude_metric)
            count = count + 1
            AS_metrics = list(set(AS_metrics) - set([exclude_metric]))
            X_AS_train = X_AS_train[AS_metrics]
        else:
            break

    print('According to Part 1 of AutoSpearman,', AS_metrics, 'are selected.')

    # (Part 2) Automatically select non-correlated metrics based on a Variance Inflation Factor analysis.
    print('(Part 2) Automatically select non-correlated metrics based on a Variance Inflation Factor analysis')

    # Prepare a dataframe for VIF
    X_AS_train = add_constant(X_AS_train)

    selected_features = X_AS_train.columns
    count = 1
    while True:
        # Calculate VIF scores
        vif_scores = pd.DataFrame([variance_inflation_factor(X_AS_train.values, i)
                                   for i in range(X_AS_train.shape[1])],
                                  index=X_AS_train.columns)
        # Prepare a final dataframe of VIF scores
        vif_scores.reset_index(inplace=True)
        vif_scores.columns = ['Feature', 'VIFscore']
        vif_scores = vif_scores.loc[vif_scores['Feature'] != 'const', :]
        vif_scores.sort_values(by=['VIFscore'], ascending=False, inplace=True)

        # Find features that have their VIF scores of above the threshold
        filtered_vif_scores = vif_scores[vif_scores['VIFscore'] >= VIF_threshold]

        # Terminate when there is no features with the VIF scores of above the threshold
        if len(filtered_vif_scores) == 0:
            break

        # exclude the metric with the highest VIF score
        metric_to_exclude = list(filtered_vif_scores['Feature'].head(1))[0]

        print('> Step', count, '- exclude', str(metric_to_exclude))
        count = count + 1

        selected_features = list(set(selected_features) - set([metric_to_exclude]))

        X_AS_train = X_AS_train.loc[:, selected_features]

    print('Finally, according to Part 2 of AutoSpearman,', AS_metrics, 'are selected.')
    all_cols = X_train.columns
    for col in all_cols:
        if col not in list(AS_metrics):
            all_cols = all_cols.drop(col)
    selected = all_cols
    X_train = X_train.loc[:, selected]
    return X_train


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_virtualenv():
    return get_base_prefix_compat() != sys.prefix


INSIDE_VIRTUAL_ENV = in_virtualenv()


def data_validation(data):
    """Validate that the given data format is a list of dictionary.

    Parameters
    ----------
    data : :obj:`Any`
        Data to be validated.

    Returns
    -------
    :obj:`bool`
        True: The data is a list of dictionary.\n
        False: The data is not a list of dictionary.
    """
    valid = True
    if isinstance(data, list):
        for i in range(len(data)):
            if not isinstance(data[i], dict):
                print("Data Format Error - the input data should be a list of dictionary")
                valid = False
                break
    else:
        valid = False
    return valid


def filter_rules(rules, X_explain):
    """Get rules that are actually applied to the commit

    Parameters
    ----------
    rules : :obj:`pandas.core.frame.DataFrame`
        Rules data under the column called 'rule' inside Rules DF generated by RuleFit
    X_explain : :obj:`pandas.core.frame.DataFrame`
        Features to be explained by the local RuleFit model, can be seen as X_test

    Returns
    -------
    :obj:`pandas.core.frame.DataFrame`
        A DataFrame that contains filtered rules
    """

    def eval_rule(rule, x_df):
        """Parsing pattern of one rule (a single row under the rule column inside Rules DF generated by RuleFit)

        Note. Only support names of rules (feature names) that has CamelCase pattern with or without underscore
        e.g. 'ThisIsLegal_Name', 'this_is_legal_name', 'thisislegalname', 'ThisIsLegalName'
        Name like 'this is not a legal name', consider using '_' instead of ' ' to concatenate
        """
        var_in_rule = list(set(re.findall('[a-z_*A-Z]+', rule)))
        rule = re.sub(r'\b=\b', '==', rule)
        if 'or' in var_in_rule:
            var_in_rule.remove('or')
        rule = rule.replace('&', 'and')
        eval_result_list = []
        for i in range(0, len(x_df)):
            x = x_df.iloc[[i]]
            var_dict = {}
            for var in var_in_rule:
                var_dict[var] = float(x[var])
            eval_result = eval(rule, var_dict)
            eval_result_list.append(eval_result)
        return eval_result_list

    # select rules that (1) have positive coefficient values and (2) have positive importance scores
    rules = rules[(rules['type'] == 'rule') & (rules['coef'] > 0) & (rules['importance'] > 0)]
    rules_list = list(rules['rule'])
    rule_eval_result = []
    # for each rule, check whether such rules apply to the actual instance to be explained
    # Note.
    # you may pass rules variable to eval_rule() to get the result of all rules
    # never try, but if it works please tell me.
    for r in rules_list:
        # X_explain must be a dataframe of 1 row
        py_exp_pred = eval_rule(r, X_explain)[0]
        rule_eval_result.append(py_exp_pred)
    new_col = {'is_satisfy_instance': rule_eval_result}
    df = pd.DataFrame(data=new_col)
    rules = pd.concat([rules, df], axis=1)
    # select rules that apply to the actual instance to be explained
    # Note. can't use 'is' as comparing operator because of pandas
    rules = rules.loc[rules['is_satisfy_instance'] == True]
    # sort rules according to their importance scores
    sorted_rules = rules.sort_values(by='importance', ascending=False)
    return sorted_rules


def get_dflt():
    """Obtain the default data and model

    Returns
    -------
    :obj:`dict`
        A dictionary wrapping all default data and model
    """
    this_dir, _ = os.path.split(__file__)
    path_dflt = this_dir + "/default_data/dflt_data.pkl"
    path_rf_model = this_dir + "/rf_models/jit_rf_model.pkl"
    path_explain = this_dir + "/default_data/explained_instance.pkl"
    if INSIDE_VIRTUAL_ENV:
        cwd = os.getcwd()
        path_dflt = cwd + "/tests/default_data/dflt_data.pkl"
        path_rf_model = cwd + "/tests/rf_models/jit_rf_model.pkl"
        path_explain = cwd + "/tests/default_data/explained_instance.pkl"
    with open(path_dflt, 'rb') as f:
        dflt_data = pickle.load(f)
    with open(path_rf_model, 'rb') as f:
        rf_model = pickle.load(f)
    with open(path_explain, 'rb') as f:
        explained_instance = pickle.load(f)

    full_ft_names = ['Number Of Lines Added', 'Number Of Modified Directories',
                     'Number Of Modified Subsystems', 'Entropy',
                     'Number Of Review Revisions', 'Review Timespan',
                     'Number of Self Approvals', 'Number of Past Developers',
                     'Time Since Last Modification', 'Number of Reviewrs',
                     'Relative Reviewer Experience', 'Author Awareness', 'Reviewer Awareness']
    return {'X_train': dflt_data['x_train'],
            'y_train': dflt_data['y_train'],
            'indep': dflt_data['indep'],
            'dep': dflt_data['dep'],
            'blackbox_model': rf_model,
            'X_explain': explained_instance['X_explain'],
            'y_explain': explained_instance['y_explain'],
            'full_ft_names': full_ft_names}


def id_generator(size=15, random_state=check_random_state(None)):
    """Generate unique ids for div tag which will contain the visualisation stuff from d3.

    Parameters
    ----------
    size : :obj:`int`
        An integer that specifies the length of the returned id, default = 15. Size should be ion range 1 - 30(both included)
    random_state : :obj:`np.random.RandomState`, default is None.
        A RandomState instance.

    Returns
    -------
    :obj:`str`
        A random identifier.
    """
    if not isinstance(size, int):
        size = 15
    if size <= 0 or size > 30:
        size = 15
    if not isinstance(random_state, np.random.mtrand.RandomState):
        random_state = check_random_state(None)
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))


def to_js_data(list_of_dict):
    """Transform python list to a str to be used inside the html <script><script/>

    Parameters
    ----------
    list_of_dict : :obj:`list`
        Data to be transformed.

    Returns
    -------
    :obj:`str`
        A str to represent a list of dict ending with ';'
    """
    if data_validation(list_of_dict):
        return str(list_of_dict) + ";"
    else:
        print("Data to be transformed to the javascript format is not a python list of dict, hence '[{}];' is returned")
        return '[{}];'


class PyExplainer:
    """A PyExplainer object is able to load training data and an ML model to generate human-centric explanation and
    visualisation

    Parameters
    ----------
    X_train : :obj:`pandas.core.frame.DataFrame`
        Training data X (Features)
    y_train : :obj:`pandas.core.series.Series`
        Training data y (Label)
    indep : :obj:`pandas.core.indexes.base.Index`
        independent variables (column names)
    dep : :obj:`str`
        dependent variables (column name)
    blackbox_model : :obj:`sklearn.ensemble.RandomForestClassifier`
        A global random forest model trained from sklearn
    class_label : :obj:`list`
        Classification labels, default = ['Clean', 'Defect']
    top_k_rules : :obj:`int`
        Number of top positive and negative rules to be retrieved
    full_ft_names : :obj:`list`
        A list containing full feature names inside X_train
    """

    def __init__(self,
                 X_train,
                 y_train,
                 indep,
                 dep,
                 blackbox_model,
                 class_label=['Clean', 'Defect'],
                 top_k_rules=3,
                 full_ft_names=[]):
        if isinstance(X_train, pd.core.frame.DataFrame):
            self.X_train = X_train
        else:
            print("X_train should be type 'pandas.core.frame.DataFrame'")
            raise TypeError
        if isinstance(y_train, pd.core.series.Series):
            self.y_train = y_train
        else:
            print("y_train should be type 'pandas.core.series.Series'")
            raise TypeError
        if isinstance(indep, pd.core.indexes.base.Index):
            self.indep = indep
        else:
            print("indep (feature column names) should be type 'pandas.core.indexes.base.Index'")
            raise TypeError
        if isinstance(dep, str):
            self.dep = dep
        else:
            print("dep (label column name) should be type 'str'")
            raise TypeError
        all_clf = all_estimators(type_filter="classifier")
        supported_algo = [clf[1] for clf in all_clf]
        if type(blackbox_model) in supported_algo:
            self.blackbox_model = blackbox_model
        else:
            print("The blackbox_model should be a classifier provided by sklearn)")
            raise TypeError
        if isinstance(class_label, list):
            if len(class_label) == 2:
                self.class_label = class_label
            else:
                print("class_label should be a list with length of 2")
                raise ValueError
        else:
            print("class_label should be type 'list'")
            raise TypeError
        if isinstance(top_k_rules, int):
            if top_k_rules <= 0 or top_k_rules > 15:
                print("top_k_rules should be in range 1 - 15 (both included)")
                raise ValueError
            else:
                self.top_k_rules = top_k_rules
        else:
            print("top_k_rules should be type 'int'")
            raise TypeError
        if full_ft_names:
            short_ft_names = X_train.columns.to_list()
            # length of short ft names and full ft names should be the same
            if len(short_ft_names) != len(full_ft_names):
                print("list of short feature names and list of full feature names should have the same length!")
                raise ValueError
            self.full_ft_names = dict(zip(short_ft_names, full_ft_names))
        else:
            self.full_ft_names = {}

        self.bullet_data = [{}]
        self.risk_data = [{}]
        self.bullet_output = widgets.Output(layout={'border': '3px solid black'})
        self.hbox_items = []
        self.X_explain = None
        self.y_explain = None

    def auto_spearman(self,
                      apply_to_X_train=True,
                      correlation_threshold=0.7,
                      correlation_method='spearman',
                      VIF_threshold=5):
        """An automated feature selection approach that address collinearity and multicollinearity.
        For more information, please kindly refer to the `paper <https://ieeexplore.ieee.org/document/8530020>`_.

        Parameters
        ----------
        apply_to_X_train : :obj:`bool`
            Whether to apply the selected columns to the X_train data inside PyExplainer Obj., default is True
        correlation_threshold : :obj:`float`
            Threshold value of correalation.
        correlation_method : :obj:`str`
            Method for solving the correlation between the features.
        VIF_threshold : :obj:`int`
            Threshold value of VIF score.
        """
        X_AS_train = AutoSpearman(correlation_threshold, correlation_method, VIF_threshold)
        if apply_to_X_train:
            self.set_X_train(X_AS_train)
            # if there is data of full feature names
            if self.get_full_ft_names():
                full_ft_names = self.get_full_ft_names()
                new_full_ft_names = {}
                for key in X_AS_train.columns.to_list():
                    new_full_ft_names[key] = full_ft_names[key]
                self.set_full_ft_names(new_full_ft_names)
            return print('X_train data inside PyExplainer was updated based on the selected features above')
        else:
            return X_AS_train

    def explain(self,
                X_explain,
                y_explain,
                top_k=3,
                max_rules=2000,
                max_iter=10000,
                cv=5,
                search_function='CrossoverInterpolation',
                debug=False):
        """Generate Rule Object Manually by passing X_explain and y_explain

        Parameters
        ----------
        X_explain : :obj:`pandas.core.frame.DataFrame`
            Features to be explained by the local RuleFit model, can be seen as X_test
        y_explain : :obj:`pandas.core.series.Series`
            Label to be explained by the local RuleFit model, can be seen as y_test
        top_k : :obj:`int`, default is 3
            Number of top rules to be retrieved
        max_rules : :obj:`int`, default is 10
            Number of maximum rules to be generated
        max_iter : :obj:`int`, default is 10
            Maximum number of iteration to be tuned in to the local RuleFit model
        cv : :obj:`int`, default is 5
            Cross Validation to be tuned in to the local RuleFit model
        search_function : :obj:`str`, default is 'crossoverinterpolation'
            Name of the search function to be used to generate the instance used by RuleFit.fit()
        debug : :obj:`bool`, default is False
            True for debugging mode, False otherwise.

        Returns
        -------
        :obj:`dict`
            A dict rule object including all of the data related to the local RuleFit model with the following keys, 'synthetic_data', 'synthetic_predictions', 'X_explain', 'y_explain', 'indep', 'dep', 'top_k_positive_rules', 'top_k_negative_rules'.

        Examples
        --------
        >>> from pyexplainer.pyexplainer_pyexplainer import PyExplainer
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> data = pd.read_csv('../tests/pyexplainer_test_data/activemq-5.0.0.csv', index_col = 'File')
        >>> dep = data.columns[-4]
        >>> indep = data.columns[0:(len(data.columns) - 4)]
        >>> X_train = data.loc[:, indep]
        >>> y_train = data.loc[:, dep]
        >>> blackbox_model = RandomForestClassifier(max_depth=3, random_state=0)
        >>> blackbox_model.fit(X_train, y_train)
        >>> class_label = ['Clean', 'Defect']
        >>> py_explainer = PyExplainer(X_train, y_train, indep, dep, class_label, blackbox_model)
        >>> sample_test_data = pd.read_csv('../tests/pyexplainer_test_data/activemq-5.0.0.csv', index_col = 'File')
        >>> X_test = sample_test_data.loc[:, indep]
        >>> y_test = sample_test_data.loc[:, dep]
        >>> sample_explain_index = 0
        >>> X_explain = X_test.iloc[[sample_explain_index]]
        >>> y_explain = y_test.iloc[[sample_explain_index]]
        >>> py_explainer.explain(X_explain, y_explain, search_function = 'crossoverinterpolation', top_k = 3, max_rules=30, max_iter =5, cv=5, debug = False)
        """
        # check if X_explain is a DF
        if not isinstance(X_explain, pd.core.frame.DataFrame):
            print("X_explain should be type 'pandas.core.frame.DataFrame'")
            raise TypeError
        # check if X_explain has the same num of cols as X_train
        if len(X_explain.columns) != len(self.X_train.columns):
            print("X_explain should have the same number of columns as X_train")
            raise ValueError
        # check if y_explain is a Series
        if not isinstance(y_explain, pd.core.series.Series):
            print("y_explain should be type 'pandas.core.series.Series'")
            raise TypeError
        self.set_top_k_rules(top_k)
        # Step 1 - Generate synthetic instances
        if search_function.lower() == 'crossoverinterpolation':
            synthetic_object = self.generate_instance_crossover_interpolation(X_explain, y_explain, debug=debug)
        elif search_function.lower() == 'randomperturbation':
            # This random perturbation approach to generate instances is used by LIME to gerate synthetic instances
            synthetic_object = self.generate_instance_random_perturbation(X_explain=X_explain, debug=debug)

        # Step 2 - Generate predictions of synthetic instances using the global model
        synthetic_instances = synthetic_object['synthetic_data'].loc[:, self.indep]
        synthetic_predictions = self.blackbox_model.predict(synthetic_instances)
        if 1 in synthetic_predictions and 0 in synthetic_predictions:
            one_class_problem = False
        else:
            one_class_problem = True
        if one_class_problem:
            print("Random Perturbation only generated one class for the prediction column which means\
                   Random Perturbation is not compatible with the current data.\
                   The 'Crossover and Interpolation' approach is used as the alternative.")
            synthetic_object = self.generate_instance_crossover_interpolation(X_explain, y_explain, debug=debug)
            synthetic_instances = synthetic_object['synthetic_data'].loc[:, self.indep]
            synthetic_predictions = self.blackbox_model.predict(synthetic_instances)

        if debug:
            n_defect_class = np.sum(synthetic_predictions)
            print('nDefect=', n_defect_class,
                  'from', len(synthetic_predictions))

        # Step 3 - Build a RuleFit local model with synthetic instances
        # indep_index = [list(synthetic_instances.columns).index(i) for i in self.indep]
        local_rulefit_model = RuleFit(rfmode='classify',
                                      exp_rand_tree_size=False,
                                      random_state=0,
                                      max_rules=max_rules,
                                      cv=cv,
                                      max_iter=max_iter,
                                      n_jobs=-1)
        local_rulefit_model.fit(synthetic_instances.values,
                                synthetic_predictions,
                                feature_names=self.indep)
        if debug:
            print('Constructed a RuleFit model')

        # Step 4 Get rules from theRuleFit local model
        rules = local_rulefit_model.get_rules()
        rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
        rules = rules[rules.type == 'rule'].sort_values("importance", ascending=False)
        positive_filtered_rules = filter_rules(rules, X_explain)

        # positive rules
        top_k_positive_rules = positive_filtered_rules.loc[positive_filtered_rules['coef'] > 0].sort_values(
            "importance", ascending=False).head(top_k)
        top_k_positive_rules['Class'] = self.class_label[1]
        top_k_positive_rules = positive_filtered_rules.reset_index()

        # negative rules
        top_k_negative_rules = rules.loc[rules['coef'] < 0].sort_values("importance", ascending=False).head(top_k)
        top_k_negative_rules['Class'] = self.class_label[0]

        rule_obj = {'synthetic_data': synthetic_instances,
                    'synthetic_predictions': synthetic_predictions,
                    'X_explain': X_explain,
                    'y_explain': y_explain,
                    'indep': self.indep,
                    'dep': self.dep,
                    'top_k_positive_rules': top_k_positive_rules,
                    'top_k_negative_rules': top_k_negative_rules,
                    'local_rulefit_model': local_rulefit_model}
        return rule_obj

    def generate_bullet_data(self, parsed_rule_object):
        """Generate bullet chart data (a list of dict) to be implemented with d3.js chart.

        Parameters
        ----------
        parsed_rule_object : :obj:`dict`
            Top rules parsed from Rule object.

        Returns
        -------
        :obj:`list`
            A list of dict that contains the data needed to generate a bullet chart.
        """
        X_explain = self.__get_X_explain()
        min_max_values = self.retrieve_X_explain_min_max_values()
        # Version 01 - only visualise for what to follow (Rules => Clean)
        bullet_data = []

        for i in range(len(parsed_rule_object['top_tofollow_rules'])):
            # sample data of tmp_rule
            # {'variable': 'MAJOR_COMMIT', 'lessthan': True, 'value': '1.550000011920929'}
            tmp_rule = parsed_rule_object['top_tofollow_rules'][i]
            tmp_actual_value = round(X_explain[tmp_rule['variable']][0], 2)
            separation_point = float(tmp_rule['value'])

            if tmp_actual_value < 0 or separation_point < 0:
                print("""actual value of %s < 0, currently do not support this type of rule""" % tmp_rule['variable'])

            tmp_markers = [tmp_actual_value]

            # currently, always place marker in the middle of bullet bars
            plot_min = 0
            diff_actual_min = abs(tmp_actual_value - plot_min)
            plot_max = tmp_actual_value + diff_actual_min
            if separation_point > plot_max:
                plot_max += separation_point - plot_max
                plot_max += diff_actual_min

            diff_plot_max_min = plot_max - plot_min
            tmp_subtitle_text = 'Actual = ' + str(tmp_actual_value)
            tmp_ticks = [plot_min, plot_max]

            if diff_plot_max_min <= 1:
                tmp_step = [0.01]
            elif diff_plot_max_min >= 1000:
                tmp_step = [10]
            elif diff_plot_max_min <= 3:
                tmp_step = [0.1]
            else:
                tmp_step = [1]

            bullet_total_width = 450
            tmp_start_points = [0, round((separation_point / diff_plot_max_min if diff_plot_max_min else 0)
                                         * bullet_total_width, 4)]
            tmp_widths = [round(tmp_start_points[1] - plot_min, 4), round(bullet_total_width - tmp_start_points[1], 4)]

            title_id = '#' + str(i + 1)
            var_name = str(tmp_rule['variable'])
            var_ref = var_name
            # todo - use get
            # check if there is mapping for full feature names
            if self.full_ft_names:
                # todo - use get
                full_ft_names = self.full_ft_names
                var_name = full_ft_names[var_name]
            if tmp_rule['lessthan']:
                # The rule suggest to decrease the values to less than a certain threshold
                tmp_title_text = title_id + ' The value of ' + \
                                 var_name + ' is more than ' + \
                                 str(tmp_actual_value)
                tmp_colors = ["#a6d96a", "#d7191c"]
            else:
                # lessthan == FALSE:
                # The rule suggest to increase the values to more than a certain threshold
                tmp_title_text = title_id + ' The value of ' + \
                                 var_name + ' is less than ' + \
                                 str(tmp_actual_value)
                tmp_colors = ["#d7191c", "#a6d96a"]

            bullet_data.append({
                "title": tmp_title_text,
                "subtitle": tmp_subtitle_text,
                "ticks": tmp_ticks,
                "step": tmp_step,
                "startPoints": tmp_start_points,
                "widths": tmp_widths,
                "colors": tmp_colors,
                "markers": tmp_markers,
                "varRef": var_ref,
            })
        return bullet_data

    def generate_html(self):
        """Generate d3 bullet chart html and return it as a String.

        Returns
        ----------
        :obj:`str`
            html String
        """
        this_dir, _ = os.path.split(__file__)
        with open(os.path.join(this_dir, 'css/styles.css'), encoding="utf8") as f:
            style_css = f.read()
        with open(os.path.join(this_dir, 'js/d3.min.js'), encoding="utf8") as f:
            d3_js = f.read()
        with open(os.path.join(this_dir, 'js/bullet.js'), encoding="utf8") as f:
            bullet_js = f.read()

        css_stylesheet = """
        <style>%s</style>
        """ % style_css

        d3_script = """
        <script>%s</script>
        <script>%s</script>
        """ % (d3_js, bullet_js)

        main_title = "Why this commit is predicted as defect-introducing?"
        title = """
        <div style="position: relative; top: 0; width: 100vw; left: 20vw;">
            <b>%s</b>
        </div>
        """ % main_title

        unique_id = id_generator()
        bullet_data = to_js_data(self.__get_bullet_data())

        d3_operation_script = """
        <script>

        var margin = { top: 5, right: 40, bottom: 20, left: 500 },
          width = 990 - margin.left - margin.right,
          height = 50 - margin.top - margin.bottom;

        var chart = d3.bullet().width(width).height(height);

        var bulletData = %s

        var svg = d3
          .select("#d3-target-bullet-%s")
          .selectAll("svg")
          .data(bulletData)
          .enter()
          .append("svg")
          .attr("class", "bullet")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g")
          .attr(
            "transform",
            "translate(" + margin.left + "," + margin.top + ")"
          )
          .call(chart);

        var title = svg
          .append("g")
          .style("text-anchor", "end")
          .attr("transform", "translate(-6," + height / 2 + ")");

        title
          .append("text")
          .attr("class", "title")
          .text((d) => d.title);

        title
          .append("text")
          .attr("class", "subtitle")
          .attr("dy", "1em")
          .text((d) => d.subtitle);

        </script>
        """ % (bullet_data, unique_id)

        html = """
        <!DOCTYPE html>
        <html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head>
            %s
            %s
        </head>
        <body>
            <div class="bullet-chart">
                %s
                <div class="d3-target-bullet" id="d3-target-bullet-%s" />
            </div>
            %s
        </body>
        </html>
        """ % (css_stylesheet, d3_script, title, unique_id, d3_operation_script)

        return html

    def generate_instance_crossover_interpolation(self, X_explain, y_explain, debug=False):
        """An approach to generate instance using Crossover and Interpolation

        Parameters
        ----------
        X_explain : :obj:`pandas.core.frame.DataFrame`
            X_explain (Testing Features)
        y_explain : :obj:`pandas.core.series.Series`
            y_explain (Testing Label)
        debug : :obj:`bool`
            True for debugging mode, False otherwise.

        Returns
        -------
        :obj:`dict`
            A dict with two keys 'synthetic_data' and 'sampled_class_frequency' generated via Crossover and Interpolation.
        """
        # categorical_vars = []

        X_train_i = self.X_train.copy()
        # y_train_i = self.y_train.copy()
        X_explain = X_explain.copy()
        y_explain = y_explain.copy()

        X_train_i.reset_index(inplace=True)
        X_explain.reset_index(inplace=True)
        X_train_i = X_train_i.loc[:, self.indep]
        # y_train_i = y_train_i.reset_index()[[self.dep]]

        X_explain = X_explain.loc[:, self.indep]
        y_explain = y_explain.reset_index()[[self.dep]]

        # get the global model predictions for the training set
        target_train = self.blackbox_model.predict(X_train_i)

        # class variables
        # ori_dataset = pd.concat([X_train_i.reset_index(drop=True), y_train_i], axis=1)

        # Do feature scaling for continuous data and one hot encoding for categorical data
        scaler = StandardScaler()
        trainset_normalize = X_train_i.copy()
        if debug:
            print(list(X_train_i), "columns")
        cases_normalize = X_explain.copy()

        train_objs_num = len(trainset_normalize)
        dataset = pd.concat(objs=[trainset_normalize, cases_normalize], axis=0)
        if debug:
            print(self.indep, "continuous")
            print(type(self.indep))
        dataset[self.indep] = scaler.fit_transform(dataset[self.indep])
        # dataset = pd.get_dummies(dataset, prefix_sep="__", columns=self.__categorical_vars)
        trainset_normalize = copy.copy(dataset[:train_objs_num])
        cases_normalize = copy.copy(dataset[train_objs_num:])

        # make dataframe to store similarities of the trained instances from the explained instance
        dist_df = pd.DataFrame(index=trainset_normalize.index.copy())

        width = math.sqrt(len(X_train_i.columns)) * 0.75
        # similarity
        for count, case in cases_normalize.iterrows():
            # Calculate the euclidean distance from the instance to be explained
            dist = np.linalg.norm(
                trainset_normalize.sub(np.array(case)), axis=1)
            # Convert distance to a similarity score
            similarity = np.exp(-(dist ** 2) / (2 * (width ** 2)))
            dist_df['dist'] = similarity
            dist_df['t_target'] = target_train
            # get the unique classes of the training set
            unique_classes = dist_df.t_target.unique()
            # Sort similarity scores in to descending order
            dist_df.sort_values(by=['dist'], ascending=False, inplace=True)
            # dist_df.reset_index(inplace=True)

            # Make a dataframe with top 40 elements in each class
            top_fourty_df = pd.DataFrame([])
            for clz in unique_classes:
                top_fourty_df = top_fourty_df.append(dist_df[dist_df['t_target'] == clz].head(40))
            # top_fourty_df.reset_index(inplace=True)

            # get the minimum value of the top 40 elements and return the index
            cutoff_similarity = top_fourty_df.nsmallest(1, 'dist', keep='last').index.values.astype(int)[0]

            # Get the location for the given index with the minimum similarity
            min_loc = dist_df.index.get_loc(cutoff_similarity)
            # whole neighbourhood without undersampling the majority class
            train_neigh_sampling_b = dist_df.iloc[0:min_loc + 1]
            # get the size of neighbourhood for each class
            target_details = train_neigh_sampling_b.groupby(['t_target']).size()
            if debug:
                print(target_details, "target_details")
            target_details_df = pd.DataFrame({'target': target_details.index, 'target_count': target_details.values})

            # Get the majority class and undersample
            final_neighbours_similarity_df = pd.DataFrame([])
            for index, row in target_details_df.iterrows():
                if row["target_count"] > 200:
                    filterd_class_set = train_neigh_sampling_b \
                        .loc[train_neigh_sampling_b['t_target'] == row['target']] \
                        .sample(n=200)
                    final_neighbours_similarity_df = final_neighbours_similarity_df.append(filterd_class_set)
                else:
                    filterd_class_set = train_neigh_sampling_b \
                        .loc[train_neigh_sampling_b['t_target'] == row['target']]
                    final_neighbours_similarity_df = final_neighbours_similarity_df.append(filterd_class_set)
            if debug:
                print(final_neighbours_similarity_df,
                      "final_neighbours_similarity_df")
            # Get the original training set instances which is equal to the index of the selected neighbours
            train_set_neigh = X_train_i[X_train_i.index.isin(final_neighbours_similarity_df.index)]
            if debug:
                print(train_set_neigh, "train set neigh")
            train_class_neigh = y_explain[y_explain.index.isin(final_neighbours_similarity_df.index)]
            # train_neigh_df = train_set_neigh.join(train_class_neigh)
            # class_neigh = train_class_neigh.groupby([self.dep]).size()

            new_con_df = pd.DataFrame([])
            sample_classes_arr = []
            sample_indexes_list = []

            # Generating instances using the cross-over technique
            for num in range(0, 1000):
                rand_rows = train_set_neigh.sample(2)
                sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                # similarity_both = dist_df[dist_df.index.isin(rand_rows.index)]
                sample_classes = train_class_neigh[train_class_neigh.index.isin(
                    rand_rows.index)]
                sample_classes = np.array(
                    sample_classes.to_records().view(type=np.matrix))
                sample_classes_arr.append(sample_classes[0].tolist())

                alpha_n = np.random.uniform(low=0, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                new_ins = x + (y - x) * alpha_n
                new_ins = new_ins.to_frame().T

                """
                # For Categorical Variables
                for cat in categorical_vars:
                    x_df = x.to_frame().T
                    y_df = y.to_frame().T
                    # Check similarity of x > similarity of y
                    if similarity_both.iloc[0]['dist'] > similarity_both.iloc[1]['dist']:
                        new_ins[cat] = x_df.iloc[0][cat]
                    # Check similarity of y > similarity of x
                    elif similarity_both.iloc[0]['dist'] < similarity_both.iloc[1]['dist']:
                        new_ins[cat] = y_df.iloc[0][cat]
                    else:
                        new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat]])
                """
                new_ins.name = num
                new_con_df = new_con_df.append(new_ins, ignore_index=True)

            # Generating instances using the mutation technique
            for num in range(1000, 2000):
                rand_rows = train_set_neigh.sample(3)
                sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                sample_classes = train_class_neigh[train_class_neigh.index.isin(rand_rows.index)]
                sample_classes = np.array(sample_classes.to_records().view(type=np.matrix))
                sample_classes_arr.append(sample_classes[0].tolist())
                mu_f = np.random.uniform(low=0.5, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                z = rand_rows.iloc[2]
                new_ins = x + (y - z) * mu_f
                new_ins = new_ins.to_frame().T
                """
                # For Categorical Variables get the value of the closest instance to the explained instance
                for cat in categorical_vars:
                    x_df = x.to_frame().T
                    y_df = y.to_frame().T
                    z_df = z.to_frame().T
                    new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat], z_df.iloc[0][cat]])
                """
                new_ins.name = num
                new_con_df = new_con_df.append(new_ins, ignore_index=True)

            # get the global model predictions of the generated instances and the instances in the neighbourhood
            predict_dataset = train_set_neigh.append(new_con_df, ignore_index=True)
            target = self.blackbox_model.predict(predict_dataset)
            target_df = pd.DataFrame(target)

            # neighbor_frequency = Counter(tuple(sorted(entry)) for entry in sample_classes_arr)

            new_df_case = pd.concat([predict_dataset, target_df], axis=1)
            new_df_case = np.round(new_df_case, 2)
            new_df_case.rename(columns={0: y_explain.columns[0]}, inplace=True)
            sampled_class_frequency = new_df_case.groupby([self.dep]).size()

            return {'synthetic_data': new_df_case,
                    'sampled_class_frequency': sampled_class_frequency}

    def generate_instance_random_perturbation(self, X_explain, debug=False):
        """The random perturbation approach to generate synthetic instances which is also used by LIME.

        Parameters
        ----------
        X_explain : :obj:`pandas.core.frame.DataFrame`
            X_explain (Testing Features)
        debug : :obj:`bool`
            True for debugging mode, False otherwise.

        Returns
        -------
        :obj:`dict`
            A dict with two keys 'synthetic_data' and 'sampled_class_frequency' generated via Random Perturbation.

        """
        random_seed = 0
        data_row = X_explain.loc[:, self.indep].values
        num_samples = 1000
        sampling_method = 'gaussian'
        discretizer = None
        sample_around_instance = True
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        scaler.fit(self.X_train.loc[:, self.indep])
        # distance_metric = 'euclidean'
        random_state = check_random_state(random_seed)
        is_sparse = sp.sparse.issparse(data_row)

        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix(
                (num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))

        if discretizer is None:
            instance_sample = data_row
            scale = scaler.scale_
            # mean = scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                # mean = mean[non_zero_indexes]

            if sampling_method == 'gaussian':
                data = random_state.normal(0, 1, num_samples * num_cols).reshape(num_samples, num_cols)
                data = np.array(data)

            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = random_state.normal(0, 1, num_samples * num_cols).reshape(num_samples, num_cols)
                data = np.array(data)

            if sample_around_instance:
                data = data * scale + instance_sample
            # else:
            #    data = data * scale + mean

            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples, data_row.shape[1]), dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(range(0, len(non_zero_indexes) * (num_samples + 1), len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix((data_1d, indexes, indptr), shape=(num_samples, data_row.shape[1]))

            # first_row = data_row
        # else:
        # first_row = discretizer.discretize(data_row)

        data[0] = data_row.copy()
        inverse = data.copy()

        # todo - this for-loop is for categorical columns in the future
        """ 
        for column in categorical_features:
            values = feature_values[column]
            freqs = feature_frequencies[column]
            inverse_column = random_state.choice(values, size=num_samples,
                                                 replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        """

        # if discretizer is not None:
        #    inverse[1:] = discretizer.undiscretize(inverse[1:])

        inverse[0] = data_row

        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - scaler.mean_) / scaler.scale_
            # distances = sklearn.metrics.pairwise_distances(scaled_data,
            #                                               scaled_data[0].reshape(1, -1),
            #                                               metric=distance_metric).ravel()

        new_df_case = pd.DataFrame(data=scaled_data, columns=self.indep)
        sampled_class_frequency = 0

        n_defect_class = np.sum(self.blackbox_model.predict(new_df_case.loc[:, self.indep]))

        if debug:
            print('Random seed', random_seed, 'nDefective', n_defect_class)

        return {'synthetic_data': new_df_case,
                'sampled_class_frequency': sampled_class_frequency}

    def generate_risk_data(self, X_explain):
        """Generate risk prediction and risk score to be visualised

        Parameters
        ----------
        X_explain : :obj:`pandas.core.frame.DataFrame`
            Explained Dataframe generated from RuleFit model.

        Returns
        -------
        :obj:`list`
            A list of dict that contains the data of risk prediction and risk score.
        """
        risk_pred = int(self.blackbox_model.predict(X_explain)[0])
        return [{"riskScore": [str(int(round(self.blackbox_model.predict_proba(X_explain)[0][1] * 100, 0))) + '%'],
                 "riskPred": [self.class_label[risk_pred]]
                 }]

    def get_full_ft_names(self):
        """getter of self.full_ft_names

        Returns
        ----------
        :obj:`list`
            A list of full feature names in X_train following the same order as X_train
        """
        return self.full_ft_names

    def get_risk_pred(self):
        """Retrieve the risk prediction from risk_data

        Returns
        ----------
        :obj:`str`
            A string of risk prediction
        """
        return self.__get_risk_data()[0]['riskPred'][0]

    def get_risk_score(self):
        """Retrieve the risk score from risk_data

        Returns
        ----------
        :obj:`float`
            A float of risk score
        """
        risk_score = self.__get_risk_data()[0]['riskScore'][0].strip("%")
        return float(risk_score)

    def get_top_k_rules(self):
        """Getter of top_k_rules

        Returns
        ----------
        :obj:`int`
            Number of top positive and negative rules to be retrieved
        """
        return self.top_k_rules

    def generate_progress_bar_items(self):
        """Generate items to be set into hbox (horizontal box)

        """
        progress_bar = widgets.FloatProgress(value=0,
                                             min=0,
                                             max=100,
                                             bar_style='info',
                                             layout=widgets.Layout(width='40%'),
                                             orientation='horizontal')
        left_text = widgets.Label("Risk Score: ")
        right_text = widgets.Label("0")
        self.__set_hbox_items([left_text, progress_bar, right_text, widgets.Label("%")])

    def generate_sliders(self):
        """Generate one or more slider widgets and return as a list.  Slider would be either IntSlider or FloatSlider depending on the value in the data

        Returns
        -------
        :obj:`list`
            A list of slider widgets.
        """
        slider_widgets = []
        data = self.__get_bullet_data()
        style = {'description_width': '40%'}
        layout = widgets.Layout(width='99%', height='20px')

        for d in data:
            # decide to use either IntSlider or FloatSlider
            if isinstance(d['step'], int):
                # create IntSlider obj and store it into a list
                slider = widgets.IntSlider(
                    value=d['markers'][0],
                    min=d['ticks'][0],
                    max=d['ticks'][-1],
                    step=d['step'][0],
                    description=d['title'],
                    layout=layout,
                    style=style,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d'
                )
                slider_widgets.append(slider)
            else:
                # create FloatSlider obj and store it into a list
                slider = widgets.FloatSlider(
                    value=d['markers'][0],
                    min=d['ticks'][0],
                    max=d['ticks'][-1],
                    step=d['step'][0],
                    description=d['title'],
                    layout=layout,
                    style=style,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f'
                )
                slider_widgets.append(slider)
        return slider_widgets

    def on_value_change(self, change, debug=False):
        """The callback function for the interactive slider

        Whenever the user interacts with the slider,
        If the slider is in the non-continuous update mode,
        only if the mouse click is released, this callback will be triggered.
        If the slider is in the continuous update mode (not recommended here),
        this function will be triggered continuously when the user is moving the slider.

        This callback will first clear the output of Risk Score Progress Bar and the Bullet Chart.
        Then it will call funcs to compute the new values to be visualised.
        When the computing is done, it will soon visualise the new value.

        Parameters
        ----------
        change : :obj:`dict`
            A dict that contains the former(before changing) and later(after changing) data inside the slider
        """
        # step 1 - clear the bullet chart output and risk score bar output
        bullet_out = self.bullet_output
        bullet_out.clear_output()

        # step 2 - compute new values to be visualised
        # get var changed
        bullet_data = self.__get_bullet_data()
        id = int(change['owner'].description.split(" ")[0].strip("#"))
        var_changed = bullet_data[id - 1]['varRef']
        if debug:
            new_value = change['new']
        else:
            new_value = change.new
        # modify changed var in X_explain
        X_explain = self.__get_X_explain()
        row_name = self.__get_X_explain().index[0]
        X_explain.at[row_name, var_changed] = new_value
        # modify bullet data
        bullet_data[id - 1]['markers'][0] = new_value
        self.__set_bullet_data(bullet_data)
        # generate new risk data
        self.__set_risk_data(self.generate_risk_data(X_explain))

        # step 3 - visualise new output
        # update risk score progress bar
        self.run_bar_animation()
        # update bullet chart
        with bullet_out:
            # display d3 bullet chart
            html = self.generate_html()
            display(HTML(html))

    def parse_top_rules(self, top_k_positive_rules, top_k_negative_rules):
        """Parse top k positive rules and top k negative rules given positive and negative rules as DataFrame

        Parameters
        ----------
        top_k_positive_rules : :obj:`pandas.core.frame.DataFrame`
            Top positive rules DataFrame
        top_k_negative_rules : :obj:`pandas.core.frame.DataFrame`
            Top negative rules DataFrame

        Returns
        -------
        :obj:`dict`
            A dict containing two keys, 'top_tofollow_rules' and 'top_toavoid_rules'
        """
        smaller_top_rule = min([len(top_k_positive_rules), len(top_k_negative_rules)])
        if self.get_top_k_rules() > smaller_top_rule:
            self.set_top_k_rules(smaller_top_rule)

        top_variables = []
        top_k_toavoid_rules = []
        top_k_tofollow_rules = []

        for i in range(len(top_k_positive_rules)):
            tmp_rule = (top_k_positive_rules['rule'].iloc[i])
            tmp_rule = tmp_rule.strip()
            tmp_rule = str.split(tmp_rule, '&')
            for j in tmp_rule:
                j = j.strip()
                tmp_sub_rule = str.split(j, ' ')
                tmp_variable = tmp_sub_rule[0]
                tmp_condition_variable = tmp_sub_rule[1]
                tmp_value = tmp_sub_rule[2]
                if tmp_variable not in top_variables:
                    top_variables.append(tmp_variable)
                    top_k_toavoid_rules.append({'variable': tmp_variable,
                                                'lessthan': tmp_condition_variable[0] == '<',
                                                'value': tmp_value})
                if len(top_k_toavoid_rules) == self.get_top_k_rules():
                    break
            if len(top_k_toavoid_rules) == self.get_top_k_rules():
                break

        for i in range(len(top_k_negative_rules)):
            tmp_rule = (top_k_negative_rules['rule'].iloc[i])
            tmp_rule = tmp_rule.strip()
            tmp_rule = str.split(tmp_rule, '&')
            for j in tmp_rule:
                j = j.strip()
                tmp_sub_rule = str.split(j, ' ')
                tmp_variable = tmp_sub_rule[0]
                tmp_condition_variable = tmp_sub_rule[1]
                tmp_value = tmp_sub_rule[2]
                if tmp_variable not in top_variables:
                    top_variables.append(tmp_variable)
                    top_k_tofollow_rules.append({'variable': tmp_variable,
                                                 'lessthan': tmp_condition_variable[0] == '<',
                                                 'value': tmp_value})
                if len(top_k_tofollow_rules) == self.get_top_k_rules():
                    break
            if len(top_k_tofollow_rules) == self.get_top_k_rules():
                break

        return {'top_tofollow_rules': top_k_tofollow_rules,
                'top_toavoid_rules': top_k_toavoid_rules}

    def retrieve_X_explain_min_max_values(self):
        """Retrieve the minimum and maximum value from X_train

        Returns
        -------
        :obj:`dict`
            A dict containing two keys, 'min_values' and 'max_values'
        """
        min_values = self.X_train.min()
        max_values = self.X_train.max()
        return {'min_values': min_values,
                'max_values': max_values}

    def run_bar_animation(self):
        """Run the animation of Risk Score Progress Bar

        """
        import time
        items_in_hbox = self.__get_hbox_items()
        progress_bar = items_in_hbox[1]

        risk_score = self.get_risk_score()
        risk_prediction = True
        if self.get_risk_pred().upper() == self.class_label[0].upper():
            risk_prediction = False
        if risk_prediction:
            progress_bar.style = {'bar_color': '#FA8128'}
        else:
            progress_bar.style = {'bar_color': '#00FF00'}

        # play speed of the animation
        play_speed = 1
        # progress bar animation
        # count start from the current val of the progress bar
        progress_bar.value = 0
        count = progress_bar.value
        right_text = items_in_hbox[2]
        while count < risk_score:
            progress_bar.value += play_speed  # signal to increment the progress bar
            new_progress_value = float(right_text.value) + play_speed

            if new_progress_value > risk_score:
                right_text.value = str(risk_score)
            else:
                right_text.value = str(new_progress_value)
            time.sleep(.01)
            count += play_speed
        # update the right text
        self.update_right_text(right_text)

    def set_full_ft_names(self, full_ft_names):
        """Setter of full_ft_names

        Parameters
        ----------
        full_ft_names : :obj:`list`
            A list of full feature names in X_train following the same order as X_train
        """
        self.full_ft_names = full_ft_names

    def set_top_k_rules(self, top_k_rules):
        """Setter of top_k_rules

        Parameters
        ----------
        top_k_rules : :obj:`int`
            Number of top positive and negative rules to be retrieved
        """
        if top_k_rules <= 0 or top_k_rules > 15 or isinstance(top_k_rules, int) == False:
            return print("set top_k_rules failed, top_k_rules should be int in range 1 - 15 (both included)")
        else:
            self.top_k_rules = top_k_rules

    def set_X_train(self, X_train):
        """Setter of X_train

        Parameters
        ----------
        X_train : :obj:`pandas.core.frame.DataFrame`
            X_train data
        """
        if isinstance(X_train, pd.core.frame.DataFrame):
            self.X_train = X_train
        else:
            return print("set X_train failed, X_train should be type of pandas.core.frame.DataFrame!")

    def show_visualisation(self):
        """Display items as follows,
        (1) Risk Score Progress Bar (made from ipywidgets)
        (2) Interactive Slider (made from ipywidgets)
        (3) Bullet Chart (Generated By D3.js)
        """
        # display risk score progress bar
        self.generate_progress_bar_items()
        items = self.__get_hbox_items()
        display(widgets.HBox(items))
        self.run_bar_animation()

        bullet_out = self.bullet_output
        bullet_out.clear_output()
        display(bullet_out)
        with bullet_out:
            # display d3 bullet chart
            html = self.generate_html()
            display(HTML(html))

        # display sliders
        sliders = self.generate_sliders()
        for slider in sliders:
            slider.observe(self.on_value_change, names='value')
            display(slider)

    def update_risk_score(self, risk_score):
        """Update the risk score value inside the risk_data

        Parameters
        ----------
        risk_score : :obj:`int`
            Value of risk score
        """
        risk_score = str(risk_score) + '%'
        self.__get_risk_data()[0]['riskScore'][0] = risk_score

    def update_right_text(self, right_text):
        """Update the text on the rightward side of the Risk Score Progress Bar

        Parameters
        ----------
        right_text : :obj:`widgets.Label`
            Text on the rightward side of the Risk Score Progress Bar
        """
        if isinstance(right_text, widgets.Label):
            self.__get_hbox_items()[2] = right_text
        else:
            print("The right_text to be set into hbox_items should be type 'ipywidgets.Label'")
            raise TypeError

    def visualise(self, rule_obj):
        """Given the rule object, show all of the visualisation as follows .
        (1) Risk Score Progress Bar (made from ipywidgets)
        (2) Interactive Slider (made from ipywidgets)
        (3) Bullet Chart (Generated By D3.js)

        Parameters
        ----------
        rule_obj : :obj:`dict`
            A rule dict generated either through loading the .pyobject file or the .explain(...) function

        Examples
        --------
        >>> from pyexplainer.pyexplainer_pyexplainer import PyExplainer
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> data = pd.read_csv('../tests/pyexplainer_test_data/activemq-5.0.0.csv', index_col = 'File')
        >>> dep = data.columns[-4]
        >>> indep = data.columns[0:(len(data.columns) - 4)]
        >>> X_train = data.loc[:, indep]
        >>> y_train = data.loc[:, dep]
        >>> blackbox_model = RandomForestClassifier(max_depth=3, random_state=0)
        >>> blackbox_model.fit(X_train, y_train)
        >>> class_label = ['Clean', 'Defect']
        >>> pyExp = PyExplainer(X_train, y_train, indep, dep, class_label, blackbox_model)
        >>> sample_test_data = pd.read_csv('../tests/pyexplainer_test_data/activemq-5.0.0.csv', index_col = 'File')
        >>> X_test = sample_test_data.loc[:, indep]
        >>> y_test = sample_test_data.loc[:, dep]
        >>> sample_explain_index = 0
        >>> X_explain = X_test.iloc[[sample_explain_index]]
        >>> y_explain = y_test.iloc[[sample_explain_index]]
        >>> rule_obj = pyExp.explain(X_explain, y_explain, search_function = 'CrossoverInterpolation', top_k = 3, max_rules=30, max_iter =5, cv=5, debug = False)
        >>> pyExp.visualise(rule_obj)
        """
        self.visualisation_data_setup(rule_obj)
        self.show_visualisation()

    def visualisation_data_setup(self, rule_obj):
        """Set up the data before visualising them

        Parameters
        ----------
        rule_obj : :obj:`dict`
            A rule dict generated either through loading the .pyobject file or the .explain(...) function
        """
        top_rules = self.parse_top_rules(top_k_positive_rules=rule_obj['top_k_positive_rules'],
                                         top_k_negative_rules=rule_obj['top_k_negative_rules'])
        self.__set_X_explain(rule_obj['X_explain'])
        self.__set_y_explain(rule_obj['y_explain'])
        self.__set_bullet_data(self.generate_bullet_data(top_rules))
        self.__set_risk_data(self.generate_risk_data(self.__get_X_explain()))

    def __get_bullet_data(self):
        """Getter of bullet_data

        Returns
        ----------
        :obj:`list`
            A list of dict that contains data needed by the d3 bullet chart
        """
        return self.bullet_data

    def __get_bullet_output(self):
        """Getter of bullet_output

        Returns
        ----------
        :obj:`ipywidgets.Output`
            A Output object used to wrap and locate contents of visualisation
        """
        return self.bullet_output

    def __get_hbox_items(self):
        """Getter of hbox_items

        Returns
        ----------
        :obj:`list`
            A list of dict that contains items to be in a horizontal box
        """
        return self.hbox_items

    def __get_risk_data(self):
        """Getter of risk_data

        Returns
        ----------
        :obj:`list`
            A list of dict that contains data needed by the d3 bullet chart
        """
        return self.risk_data

    def __get_X_explain(self):
        """Getter of X_explain

        Returns
        ----------
        :obj:`pandas.core.frame.DataFrame`
            An explained DataFrame containing the features
        """
        return self.X_explain

    def __get_y_explain(self):
        """Getter of y_explain

        Returns
        ----------
        :obj:`pandas.core.series.Series`
             An explained DataFrame containing the label
        """
        return self.y_explain

    def __set_bullet_data(self, bullet_data):
        """Setter of bullet_data

        Parameters
        ----------
        bullet_data : :obj:`list`
            A list of dict that contains data needed by the d3 bullet chart
        """
        if data_validation(bullet_data):
            self.bullet_data = bullet_data
        else:
            print('bullet_data is not in the format of python list of dict')
            raise ValueError

    def __set_bullet_output(self, bullet_output):
        """Setter of bullet_output

        Parameters
        ----------
        bullet_output : :obj:`widgets.Output`
            A Output object used to wrap and locate contents of visualisation
        """
        if isinstance(bullet_output, widgets.Output):
            self.bullet_output = bullet_output
        else:
            print("bullet_output should be type 'ipywidgets.Output'")
            raise TypeError

    def __set_hbox_items(self, hbox_items):
        """Setter of hbox_items

        Parameters
        ----------
        hbox_items : :obj:`list`
            A list of dict that contains items to be in a horizontal box
        """
        if len(hbox_items) == 4:
            if isinstance(hbox_items[0], widgets.Label) and isinstance(hbox_items[1], widgets.FloatProgress) \
                    and isinstance(hbox_items[2], widgets.Label) and isinstance(hbox_items[3], widgets.Label):
                self.hbox_items = hbox_items
            else:
                print("""hbox_items should be in the format of '[widgets.Label, widgets.FloatProgress, widgets.Label, 
                                  widgets.Label]'""")
                raise TypeError
        else:
            print("""hbox_items should be in the format of '[widgets.Label, widgets.FloatProgress, widgets.Label, 
                  widgets.Label]'""")
            raise TypeError

    def __set_risk_data(self, risk_data):
        """Setter of risk_data

        Parameters
        ----------
        risk_data : :obj:`list`
            A list of dict that contains risk prediction and risk score info
        """
        if data_validation(risk_data):
            self.risk_data = risk_data
        else:
            print('risk_data is not in the format of python list of dict')
            raise ValueError

    def __set_X_explain(self, X_explain):
        """Setter of X_explain

        Parameters
        ----------
        X_explain : :obj:`pandas.core.frame.DataFrame`
            An explained DataFrame containing feature cols
        """
        if isinstance(X_explain, pd.core.frame.DataFrame):
            self.X_explain = X_explain
        else:
            print("X_explain should be type 'pandas.core.frame.DataFrame'")
            raise TypeError

    def __set_y_explain(self, y_explain):
        """Setter of y_explain

        Parameters
        ----------
        y_explain : :obj:`pandas.core.series.Series`
            An explained DataFrame containing label col
        """
        if isinstance(y_explain, pd.core.series.Series):
            self.y_explain = y_explain
        else:
            print("y_explain should be type 'pandas.core.series.Series'")
            raise TypeError
