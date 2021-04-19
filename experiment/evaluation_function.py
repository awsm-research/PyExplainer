from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import auc, roc_auc_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns

from my_util import *
from lime.lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt

import os, pickle, time, re, sys, operator
from datetime import datetime
from collections import Counter

sys.path.append(os.path.abspath('../'))
from pyexplainer.pyexplainer_pyexplainer import *

flip_sign_dict = {
    '<': '>=',
    '>': '<=',
    '=': '!=',
    '>=': '<',
    '<=': '>',
    '!=': '=='
}

'''
    input: rule (str)
'''
def flip_rule(rule):
    rule = re.sub(r'\b=\b',' = ',rule) # for LIME
    found_rule = re.findall('.* <=? [a-zA-Z]+ <=? .*', rule) # for LIME
    ret = ''
    
    if len(found_rule) > 0:
        found_rule = found_rule[0]
    
        var_in_rule = re.findall('[a-zA-Z]+',found_rule)
        var_in_rule = var_in_rule[0]
        
        splitted_rule = found_rule.split(var_in_rule)
        splitted_rule[0] = splitted_rule[0] + var_in_rule # for left side
        splitted_rule[1] = var_in_rule + splitted_rule[1] # for right side
        combined_rule = splitted_rule[0] + ' or ' + splitted_rule[1]
        ret = flip_rule(combined_rule)
        
    else:
        for tok in rule.split():
            if tok in flip_sign_dict:
                ret = ret + flip_sign_dict[tok] + ' '
            else:
                ret = ret + tok + ' '
    return ret

def get_rule_str_of_rulefit(local_rulefit_model):
    rule_df = local_rulefit_model.get_rules()
    top_k = 5
    top_k_positive_rules = rule_df[(rule_df.coef > 0) & (rule_df.type=='rule')].sort_values("importance", ascending=False).head(top_k)

    the_best_defective_rule_str = list(top_k_positive_rules['rule'])[0]
    
    return the_best_defective_rule_str

def aggregate_list(l):
    return np.mean(l), np.median(l)

def prepare_data_for_testing(proj_name, global_model_name = 'RF'):
    global_model_name = global_model_name.upper()
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    correctly_predict_df = pd.read_csv(dump_dataframe_dir+proj_name+'_'+global_model_name+'_correctly_predict_as_defective.csv')
    correctly_predict_df = correctly_predict_df.set_index('commit_id')

    dep = 'defect'
    indep = correctly_predict_df.columns[:-3] # exclude the last 3 columns

    feature_df = correctly_predict_df.loc[:, indep]
    
    return global_model, correctly_predict_df, indep, dep, feature_df
    
def get_prediction_result_df(proj_name, global_model_name):
    global_model_name = global_model_name.upper()
    if global_model_name not in ['RF','LR']:
        print('wrong global model name. the global model name must be RF or LR')
        return
    
    prediction_df_dir = dump_dataframe_dir+proj_name+'_'+global_model_name+'_prediction_result.csv'
    prediction_df = pd.read_csv(prediction_df_dir)
    prediction_df = prediction_df.set_index('commit_id')
    
    return prediction_df

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort/100)*result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent =  result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['defect']==True]
    recall_k_percent_effort = len(buggy_commit)/float(len(real_buggy_commits))
    
    return recall_k_percent_effort

def eval_global_model(proj_name, prediction_df):
    ## since ld metric in openstack is removed by using autospearman, so this code is needed
    ## but this is not problem for qt
    
    if proj_name == 'openstack':
        x_train_original, x_test_original = prepare_data_all_metrics(proj_name, mode='all')
        prediction_df = prediction_df.copy()
        prediction_df['ld'] = list(x_test_original['ld'])
        
    prediction_df = prediction_df[['la','ld', 'pred', 'defective_prob' ,'defect']]
    prediction_df['LOC'] = prediction_df['la']+prediction_df['ld']
    
    prediction_df['defect_density'] = prediction_df['defective_prob']/prediction_df['LOC']
    prediction_df['actual_defect_density'] = prediction_df['defect']/prediction_df['LOC'] #defect density
    
    prediction_df = prediction_df.fillna(0)
    prediction_df = prediction_df.replace(np.inf, 0)
    
    prediction_df = prediction_df.sort_values(by='defect_density',ascending=False)
    
    actual_result_df = prediction_df.sort_values(by='actual_defect_density',ascending=False)
    actual_worst_result_df = prediction_df.sort_values(by='actual_defect_density',ascending=True)

    prediction_df['cum_LOC'] = prediction_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = prediction_df[prediction_df['defect'] == True]
    
    
    AUC = roc_auc_score(prediction_df['defect'], prediction_df['defective_prob'])
    f1 = f1_score(prediction_df['defect'], prediction_df['pred'])
    
    ifa = real_buggy_commits.iloc[0]['cum_LOC']

    cum_LOC_20_percent = 0.2*prediction_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = prediction_df[prediction_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['defect']==True]
    recall_20_percent_effort = len(buggy_commit)/float(len(real_buggy_commits))
    
    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []
    
    for percent_effort in np.arange(10,101,10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, prediction_df, real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df, real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df, real_buggy_commits)
        
        percent_effort_list.append(percent_effort/100)
        
        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) - 
                 auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    print('AUC: {}, F1: {}, IFA: {}, Recall@20%Effort: {}, Popt: {}'.format(AUC,f1,ifa,recall_20_percent_effort,p_opt))
    print(classification_report(prediction_df['defect'], prediction_df['pred']))

def get_global_model_evaluation_result(proj_name):
    print('RF global model result')
    rf_prediction_df = get_prediction_result_df(proj_name, 'rf')
    eval_global_model(proj_name, rf_prediction_df)

    print('-'*100)
    
    print('LR global model result')
    lr_prediction_df = get_prediction_result_df(proj_name, 'lr')
    eval_global_model(proj_name, lr_prediction_df)
    
def rq1_eval(proj_name, global_model_name):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    all_eval_result = pd.DataFrame()
    
    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(exp_dir+proj_name+'_'+global_model_name+'_all_explainer_'+row_index+'.pkl','rb'))
        py_exp = exp_obj['pyExplainer']
        lime_exp = exp_obj['LIME']

        # this data can be used for both local and global model
        py_exp_synthetic_data = py_exp['synthetic_data'].values
        # this data can be used with global model only
        lime_exp_synthetic_data = lime_exp['synthetic_instance_for_global_model']
        
        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']

        py_exp_global_pred = global_model.predict(py_exp_synthetic_data)

        lime_exp_global_pred = global_model.predict(lime_exp_synthetic_data)

        py_exp_dist = euclidean_distances(X_explain.values, py_exp_synthetic_data)
        lime_dist = euclidean_distances(X_explain.values, lime_exp_synthetic_data)

        py_exp_dist_mean, py_exp_dist_med = aggregate_list(py_exp_dist)
        lime_exp_dist_mean, lime_exp_dist_med = aggregate_list(lime_dist)

        py_exp_serie = pd.Series(data=[proj_name, row_index, 'pyExplainer',
                                       py_exp_dist_med])
        lime_exp_serie = pd.Series(data=[proj_name, row_index, 'LIME',
                                         lime_exp_dist_med])
        
        all_eval_result = all_eval_result.append(py_exp_serie,ignore_index=True)
        all_eval_result = all_eval_result.append(lime_exp_serie, ignore_index=True)
        
    all_eval_result.columns =['project', 'commit id', 'method', 'euc_dist_med']
    
    all_eval_result.to_csv(result_dir+'RQ1_'+proj_name+'_'+global_model_name+'.csv',index=False)
    print('finished RQ1 of',proj_name,', globla model is',global_model_name)
    
def show_rq1_eval_result():
    openstack_rf = pd.read_csv('./eval_result/RQ1_openstack_RF.csv')
    qt_rf = pd.read_csv('./eval_result/RQ1_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv('./eval_result/RQ1_openstack_LR.csv')
    qt_lr = pd.read_csv('./eval_result/RQ1_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    fig, axs = plt.subplots(1,2, figsize=(10,6))

    axs[0].set_title('RF')
    axs[1].set_title('LR')
    
    axs[0].set(ylim=(0, 5000))
    axs[1].set(ylim=(0, 5000))
    
    sns.boxplot(data=result_rf, x='project', y='euc_dist_med', hue='method', ax=axs[0])
    sns.boxplot(data=result_lr, x='project', y='euc_dist_med', hue='method', ax=axs[1])
    
    plt.show()

    display(all_result.groupby(['global_model', 'project', 'method']).describe())
    
    result_rf.to_csv('./eval_result/RQ1_RF.csv',index=False)
    result_lr.to_csv('./eval_result/RQ1_LR.csv',index=False)
    all_result.to_csv('./eval_result/RQ1.csv',index=False)
    
    fig.savefig('./eval_result/figures/RQ1.png')
    
def rq2_eval(proj_name, global_model_name):
    global_model_name = global_model_name.upper()
    
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    all_eval_result = pd.DataFrame()
    
    pyexp_label, pyexp_prob = [],[]
    lime_label, lime_prob = [],[]
    
    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(pyExp_dir+proj_name+'_'+global_model_name+'_all_explainer_'+row_index+'.pkl','rb'))
        py_exp = exp_obj['pyExplainer']
        lime_exp = exp_obj['LIME']

        # this data can be used for both local and global model
        py_exp_synthetic_data = py_exp['synthetic_data'].values
        # this data can be used with global model only
        lime_exp_synthetic_data = lime_exp['synthetic_instance_for_global_model']
        # this data can be used with local model only
        lime_exp_synthetic_data_local = lime_exp['synthetic_instance_for_lobal_model']
        
        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']

        py_exp_global_pred = global_model.predict(py_exp_synthetic_data) 
        py_exp_local_prob = py_exp_local_model.predict_proba(py_exp_synthetic_data)[:,1]
        py_exp_local_pred = py_exp_local_model.predict(py_exp_synthetic_data)

        lime_exp_global_pred = global_model.predict(lime_exp_synthetic_data)
        lime_exp_local_prob = lime_exp_local_model.predict(lime_exp_synthetic_data_local)
        lime_exp_local_pred = np.round(lime_exp_local_prob)
        
        pyexp_label.extend(list(py_exp_global_pred))
        pyexp_prob.extend(list(py_exp_local_prob))
        
        lime_label.extend(list(lime_exp_global_pred))
        lime_prob.extend(list(lime_exp_local_prob))
        
        
        py_exp_auc = roc_auc_score(py_exp_global_pred, py_exp_local_prob)
        py_exp_f1 = f1_score(py_exp_global_pred, py_exp_local_pred)
        
        lime_auc = roc_auc_score(lime_exp_global_pred, lime_exp_local_prob)
        lime_f1 = f1_score(lime_exp_global_pred, lime_exp_local_pred)

        py_exp_serie = pd.Series(data=[proj_name, row_index, 'pyExplainer',
                                        py_exp_auc, py_exp_f1])
        lime_exp_serie = pd.Series(data=[proj_name, row_index, 'LIME',
                                           lime_auc, lime_f1])
        
        all_eval_result = all_eval_result.append(py_exp_serie,ignore_index=True)
        all_eval_result = all_eval_result.append(lime_exp_serie, ignore_index=True)

    pred_df = pd.DataFrame()
    
    all_tech = ['pyExplainer']*len(pyexp_label) + ['LIME']*len(lime_label)
    
    pred_df['technique'] = all_tech
    pred_df['label'] = pyexp_label+lime_label
    pred_df['prob'] = pyexp_prob+lime_prob
    pred_df['project'] = proj_name
    
    all_eval_result.columns = ['project', 'commit id', 'method', 'AUC', 'F1']

    all_eval_result.to_csv(result_dir+'RQ2_'+proj_name+'_'+global_model_name+'_global_vs_local_synt_pred.csv',index=False)
    pred_df.to_csv(result_dir+'RQ2_'+proj_name+'_'+global_model_name+'_probability_distribution.csv',index=False)
    print('finished RQ2 of',proj_name)
    
def show_rq2_eval_result():
    openstack_rf = pd.read_csv('./eval_result/RQ2_openstack_RF_global_vs_local_synt_pred.csv')
    qt_rf = pd.read_csv('./eval_result/RQ2_qt_RF_global_vs_local_synt_pred.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv('./eval_result/RQ2_openstack_LR_global_vs_local_synt_pred.csv')
    qt_lr = pd.read_csv('./eval_result/RQ2_qt_LR_global_vs_local_synt_pred.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    fig, axs = plt.subplots(2,2, figsize=(10,10))

    axs[0,0].set_title('RF')
    axs[0,1].set_title('LR')
    
    axs[0,0].set_ylim([0, 1])
    axs[0,1].set_ylim([0, 1]) 
    axs[1,0].set_ylim([0, 1])
    axs[1,1].set_ylim([0, 1])

    sns.boxplot(data=result_rf, x='project', y='AUC', hue='method', ax=axs[0,0])
    sns.boxplot(data=result_rf, x='project', y='F1', hue='method', ax=axs[1,0])
    sns.boxplot(data=result_lr, x='project', y='AUC', hue='method', ax=axs[0,1])
    sns.boxplot(data=result_lr, x='project', y='F1', hue='method', ax=axs[1,1])

    plt.show()

    pd.set_option("max_columns", 40)
    display(all_result.groupby(['global_model', 'project','method']).describe())
    
    result_rf.to_csv('./eval_result/RQ2_RF_prediction.csv',index=False)
    result_lr.to_csv('./eval_result/RQ2_LR_prediction.csv',index=False)
    all_result.to_csv('./eval_result/RQ2_prediction.csv',index=False)
    
    fig.savefig('./eval_result/figures/RQ2_prediction.png')

def show_rq2_prob_distribution():
    
    d = {True: 'DEFECT', False: 'CLEAN'}

    openstack_rf = pd.read_csv('./eval_result/RQ2_openstack_RF_probability_distribution.csv')
    qt_rf = pd.read_csv('./eval_result/RQ2_qt_RF_probability_distribution.csv')
    
    mask = openstack_rf.applymap(type) != bool
    openstack_rf = openstack_rf.where(mask, openstack_rf.replace(d))
    qt_rf = qt_rf.where(mask, qt_rf.replace(d))
    
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv('./eval_result/RQ2_openstack_LR_probability_distribution.csv')
    qt_lr = pd.read_csv('./eval_result/RQ2_qt_LR_probability_distribution.csv')
    
    openstack_lr = openstack_lr.where(mask, openstack_lr.replace(d))
    qt_lr = qt_lr.where(mask, qt_lr.replace(d))
    
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    fig, axs = plt.subplots(2,2, figsize=(10,10))

    
    axs[0,0].set_ylim([0, 1])
    axs[0,1].set_ylim([0, 1]) 
    axs[1,0].set_ylim([0, 1])
    axs[1,1].set_ylim([0, 1])
    
    sns.boxplot(data=openstack_rf, x='technique', y='prob', hue='label' , ax=axs[0,0])
    sns.boxplot(data=qt_rf,  x='technique', y='prob', hue='label' , ax=axs[1,0])
    sns.boxplot(data=openstack_lr,  x='technique', y='prob', hue='label' , ax=axs[0,1])
    sns.boxplot(data=qt_lr,  x='technique', y='prob', hue='label' , ax=axs[1,1], palette=['darkorange','royalblue'])
    
    axs[0,0].axhline(0.5, ls='--')
    axs[0,1].axhline(0.5, ls='--')
    axs[1,0].axhline(0.5, ls='--')
    axs[1,1].axhline(0.5, ls='--')
    
    cols = ['Global model: RF', 'Global model: LR']
    rows = ['Openstack','Qt']

    plt.setp(axs.flat, xlabel='Technique', ylabel='Probability')

    pad = 5 # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    plt.show()
    
    result_rf.to_csv('./eval_result/RQ2_RF_prediction_prob.csv',index=False)
    result_lr.to_csv('./eval_result/RQ2_LR_prediction_prob.csv',index=False)
    all_result.to_csv('./eval_result/RQ2_prediction_prob.csv',index=False)
    
    fig.savefig('./eval_result/figures/RQ2_prediction_prob.png')
    
def eval_rule(rule, x_df):
    var_in_rule = list(set(re.findall('[a-zA-Z]+', rule)))
    
    rule = re.sub(r'\b=\b','==',rule)
    if 'or' in var_in_rule:
        var_in_rule.remove('or')
        
    rule = rule.replace('&','and')
    
    eval_result_list = []

    for i in range(0,len(x_df)):
        x = x_df.iloc[[i]]
        col = x.columns
        var_dict = {}

        for var in var_in_rule:
            var_dict[var] = float(x[var])
        
        eval_result = eval(rule,var_dict)
        eval_result_list.append(eval_result)

        
    return eval_result_list

def summarize_rule_eval_result(rule_str, x_df):
    all_eval_result = eval_rule(rule_str, x_df)
    all_eval_result = np.array(all_eval_result).astype(bool)

    return all_eval_result

def rq3_eval(proj_name, global_model_name):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    x_test, y_test = prepare_data(proj_name, mode = 'test')
    
    rq3_eval_result = pd.DataFrame() # for train data

    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(exp_dir+proj_name+'_'+global_model_name+'_all_explainer_'+row_index+'.pkl','rb'))
        py_exp = exp_obj['pyExplainer']
        lime_exp = exp_obj['LIME']

        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']
        
        py_exp_the_best_defective_rule_str = get_rule_str_of_rulefit(py_exp_local_model)
        
        lime_the_best_defective_rule_str = lime_exp['rule'].as_list()[0][0]

        py_exp_pred = eval_rule(py_exp_the_best_defective_rule_str, X_explain)[0]
        lime_pred = eval_rule(lime_the_best_defective_rule_str, X_explain)[0]

        if py_exp_pred:
            py_exp_the_best_defective_rule_str = flip_rule(py_exp_the_best_defective_rule_str)
            py_exp_rule_eval = summarize_rule_eval_result(py_exp_the_best_defective_rule_str, x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, py_exp_rule_eval, labels=[1,0]).ravel()
            tp_rate = tp/(tp+fn)
            tn_rate = tn/(tn+fp)
            
            py_exp_serie_test = pd.Series(data=[proj_name, row_index, 'pyExplainer',tp_rate,tn_rate])
        
            rq3_eval_result = rq3_eval_result.append(py_exp_serie_test,ignore_index=True)
        
        if lime_pred:
            lime_the_best_defective_rule_str = flip_rule(lime_the_best_defective_rule_str)
            lime_rule_eval = summarize_rule_eval_result(lime_the_best_defective_rule_str, x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, lime_rule_eval, labels=[1,0]).ravel()
            tp_rate = tp/(tp+fn)
            tn_rate = tn/(tn+fp)

            lime_serie_test = pd.Series(data=[proj_name, row_index, 'LIME',tp_rate,tn_rate])
        
            rq3_eval_result = rq3_eval_result.append(lime_serie_test, ignore_index=True)

        print('finished {} from {} commits'.format(str(i+1),len(feature_df)))
        
        
    rq3_eval_result.columns = ['project', 'commit id', 'method', 'true_positive_rate','true_negative_rate']
    
    rq3_eval_result.to_csv(result_dir+'RQ3_'+proj_name+'_'+global_model_name+'.csv',index=False)
    print('finished RQ3 of',proj_name)
    
def show_rq3_eval_result():

    openstack_rf = pd.read_csv('./eval_result/RQ3_openstack_RF.csv')
    qt_rf = pd.read_csv('./eval_result/RQ3_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv('./eval_result/RQ3_openstack_LR.csv')
    qt_lr = pd.read_csv('./eval_result/RQ3_qt_LR.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    my_pal = ['darkorange','blue']

    fig, axs = plt.subplots(2,2, figsize=(8,8))

    axs[0,0].set_title('RF')
    axs[0,1].set_title('LR')

    # plt.ylim(0,100)
    sns.boxplot(data=result_rf, x='project', y='true_positive_rate', hue='method', ax=axs[0,0], palette=['darkorange','royalblue'])
    sns.boxplot(data=result_rf, x='project', y='true_negative_rate', hue='method', ax=axs[1,0], palette=['darkorange','royalblue'])
    sns.boxplot(data=result_lr, x='project', y='true_positive_rate', hue='method', ax=axs[0,1], palette=['royalblue','darkorange'])
    sns.boxplot(data=result_lr, x='project', y='true_negative_rate', hue='method', ax=axs[1,1], palette=['royalblue','darkorange'])

    plt.show()
    
    display(all_result[['global_model', 'project','method', 'true_positive_rate','true_negative_rate']].groupby(['global_model','project','method']).describe())
    
    result_rf.to_csv('./eval_result/RQ3_RF.csv',index=False)
    result_lr.to_csv('./eval_result/RQ3_LR.csv',index=False)
    all_result.to_csv('./eval_result/RQ3.csv',index=False)
    
    fig.savefig('./eval_result/figures/RQ3.png')
    
    