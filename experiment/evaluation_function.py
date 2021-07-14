from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import auc, roc_auc_score, f1_score, confusion_matrix, classification_report, recall_score
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


data_path = './dataset/'
result_dir = './eval_result/'
dump_dataframe_dir = './prediction_result/'
exp_dir = './explainer_object/'

fig_dir = result_dir+'figures/'

flip_sign_dict = {
    '<': '>=',
    '>': '<=',
    '=': '!=',
    '>=': '<',
    '<=': '>',
    '!=': '=='
}

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def get_rule_str_of_rulefit(local_model, X_explain):
    rules = local_model.get_rules()
    rules = rules[(rules['type']=='rule') & (rules['coef'] > 0) & (rules['importance'] > 0)]
    rules_list = list(rules['rule'])
    
    rule_eval_result = []
    
    for r in rules_list:
        py_exp_pred = eval_rule(r, X_explain)[0]
        rule_eval_result.append(py_exp_pred)
          
    rules['is_satisfy_instance'] = rule_eval_result

    rules = rules[rules['is_satisfy_instance']==True]

    rules = rules.sort_values(by='importance', ascending=False)
    
    rule_str = rules.iloc[0]['rule']
    
    return rule_str

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

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_'+row_index+'.pkl'),'rb'))
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
    openstack_rf = pd.read_csv(result_dir+'RQ1_openstack_RF.csv')
    qt_rf = pd.read_csv(result_dir+'RQ1_qt_RF.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ1_openstack_LR.csv')
    qt_lr = pd.read_csv(result_dir+'RQ1_qt_LR.csv')
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
    
    all_result.to_csv(result_dir+'/RQ1.csv',index=False)
    
    fig.savefig(fig_dir+'RQ1.png')
    
def rq2_eval(proj_name, global_model_name):
    global_model_name = global_model_name.upper()
    
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    all_eval_result = pd.DataFrame()
    
    pyexp_label, pyexp_prob = [],[]
    lime_label, lime_prob = [],[]
    
    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_'+row_index+'.pkl'),'rb'))
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
    openstack_rf = pd.read_csv(result_dir+'RQ2_openstack_RF_global_vs_local_synt_pred.csv')
    qt_rf = pd.read_csv(result_dir+'RQ2_qt_RF_global_vs_local_synt_pred.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'/RQ2_openstack_LR_global_vs_local_synt_pred.csv')
    qt_lr = pd.read_csv(result_dir+'/RQ2_qt_LR_global_vs_local_synt_pred.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    
    openstack_result = all_result[all_result['project']=='openstack']
    qt_result = all_result[all_result['project']=='qt']

    fig, axs = plt.subplots(2,2, figsize=(10,10))

    axs[0,0].set_title('Openstack')
    axs[0,1].set_title('Qt')
    
    axs[0,0].set_ylim([0, 1])
    axs[0,1].set_ylim([0, 1]) 
    axs[1,0].set_ylim([0, 1])
    axs[1,1].set_ylim([0, 1])

#     rows = ['AUC', 'F1']
    cols = ['Openstack','Qt']

#     plt.setp(axs.flat, xlabel='Technique', ylabel='Probability')

    pad = 5 # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

#     for ax, row in zip(axs[:,0], rows):
#         ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
#                     xycoords=ax.yaxis.label, textcoords='offset points',
#                     size='large', ha='right', va='center')
        
    sns.boxplot(data=openstack_result, x='global_model', y='AUC', hue='method', ax=axs[0,0])
    sns.boxplot(data=openstack_result, x='global_model', y='F1', hue='method', ax=axs[1,0])
    sns.boxplot(data=qt_result, x='global_model', y='AUC', hue='method', ax=axs[0,1])
    sns.boxplot(data=qt_result, x='global_model', y='F1', hue='method', ax=axs[1,1])

    plt.show()
    
    all_result.to_csv(result_dir+'RQ2_prediction.csv',index=False)
    
    fig.savefig(fig_dir+'RQ2_prediction.png')

def show_rq2_prob_distribution():
    
    d = {True: 'DEFECT', False: 'CLEAN'}

    openstack_rf = pd.read_csv(result_dir+'RQ2_openstack_RF_probability_distribution.csv')
    qt_rf = pd.read_csv(result_dir+'RQ2_qt_RF_probability_distribution.csv')
    
    mask = openstack_rf.applymap(type) != bool
    openstack_rf = openstack_rf.where(mask, openstack_rf.replace(d))
    qt_rf = qt_rf.where(mask, qt_rf.replace(d))
    
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ2_openstack_LR_probability_distribution.csv')
    qt_lr = pd.read_csv(result_dir+'RQ2_qt_LR_probability_distribution.csv')
    
    openstack_lr = openstack_lr.where(mask, openstack_lr.replace(d))
    qt_lr = qt_lr.where(mask, qt_lr.replace(d))
    
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])

    pyexp_openstack_result = all_result[(all_result['project']=='openstack') & (all_result['technique']=='pyExplainer')]
    pyexp_qt_result = all_result[(all_result['project']=='qt') & (all_result['technique']=='pyExplainer')]
    lime_openstack_result = all_result[(all_result['project']=='openstack') & (all_result['technique']=='LIME')]
    lime_qt_result = all_result[(all_result['project']=='qt') & (all_result['technique']=='LIME')]
    
#     qt_result = all_result[all_result['project']=='qt']
    
    fig, axs = plt.subplots(2,2, figsize=(10,10))

    axs[0,0].set_ylim([0, 1])
    axs[0,1].set_ylim([0, 1]) 
    axs[1,0].set_ylim([0, 1])
    axs[1,1].set_ylim([0, 1])
    
    sns.boxplot(data=pyexp_openstack_result, x='global_model', y='prob', hue='label' , ax=axs[0,0])
    sns.boxplot(data=lime_openstack_result,  x='global_model', y='prob', hue='label' , ax=axs[1,0])
    sns.boxplot(data=pyexp_qt_result,  x='global_model', y='prob', hue='label' , ax=axs[0,1])
    sns.boxplot(data=lime_qt_result,  x='global_model', y='prob', hue='label' , ax=axs[1,1], palette=['darkorange','royalblue'])
    
    axs[0,0].axhline(0.5, ls='--')
    axs[0,1].axhline(0.5, ls='--')
    axs[1,0].axhline(0.5, ls='--')
    axs[1,1].axhline(0.5, ls='--')
    
    rows = ['PyExplainer', 'LIME']
    cols = ['Openstack','Qt']

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
    
    all_result.to_csv(result_dir+'RQ2_prediction_prob.csv',index=False)
    
    fig.savefig(fig_dir+'RQ2_prediction_prob.png')
    
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

    rq3_explanation_result = pd.DataFrame()
    
    pyexp_guidance_result_list = []
    lime_guidance_result_df = pd.DataFrame()
    
    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_'+row_index+'.pkl'),'rb'))
        py_exp = exp_obj['pyExplainer']
        lime_exp = exp_obj['LIME']

        # load local models
        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']
        
        # generate explanations                
        py_exp_the_best_defective_rule_str = get_rule_str_of_rulefit(py_exp_local_model, X_explain)
        lime_the_best_defective_rule_str = lime_exp['rule'].as_list()[0][0]

        # check whether explanations apply to the instance to be explained
        py_exp_pred = eval_rule(py_exp_the_best_defective_rule_str, X_explain)[0]
        lime_pred = eval_rule(lime_the_best_defective_rule_str, X_explain)[0]

        condition_list = py_exp_the_best_defective_rule_str.split('&')

        # for explanations
        for condition in condition_list:
            condition = condition.strip()

            py_exp_rule_eval = summarize_rule_eval_result(condition, x_test)

            rule_rec = recall_score(y_test, py_exp_rule_eval)

            py_exp_serie_test = pd.Series(data=[proj_name, row_index, 'pyExplainer',global_model_name, condition, rule_rec])
            rq3_explanation_result = rq3_explanation_result.append(py_exp_serie_test,ignore_index=True)

        # PyExp END
        
        # LIME START
        lime_rule_eval = summarize_rule_eval_result(lime_the_best_defective_rule_str, x_test)

        rule_rec = recall_score(y_test, lime_rule_eval)

        lime_serie_test = pd.Series(data=[proj_name, row_index, 'LIME',global_model_name, lime_the_best_defective_rule_str, rule_rec])
        rq3_explanation_result = rq3_explanation_result.append(lime_serie_test,ignore_index=True)
            
        print('finished {} from {} commits'.format(str(i+1),len(feature_df)))

    rq3_explanation_result.columns = ['project','commit_id','method','global_model','explanation','recall']
    rq3_explanation_result.to_csv(result_dir+'RQ3_'+proj_name+'_'+global_model_name+'_explanation_eval_split_rulefit_condition.csv',
                                  index=False)
    
def get_percent_unique_explanation(proj_name, global_model_name, explanation_list):
    
    print('project: {}, JIT model: {}'.format(proj_name, global_model_name))
    total_exp = len(explanation_list)
    total_unique_exp = len(set(explanation_list))
    percent_unique = (total_unique_exp/total_exp)*100
    
    count_exp = Counter(explanation_list)
    max_exp_count = max(list(count_exp.values()))
    percent_dup_explanation = (max_exp_count/total_exp)*100
    
    print('% unique explanation is',round(percent_unique,2))
    print('% duplicate explanation is', round(percent_dup_explanation))
    
    print('-'*100)
    
def show_rq3_eval_result():

    openstack_rf = pd.read_csv(result_dir+'RQ3_openstack_RF_explanation_eval_split_rulefit_condition.csv')
    qt_rf = pd.read_csv(result_dir+'RQ3_qt_RF_explanation_eval_split_rulefit_condition.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])
    result_rf['global_model'] = 'RF'
    
    openstack_lr = pd.read_csv(result_dir+'RQ3_openstack_LR_explanation_eval_split_rulefit_condition.csv')
    qt_lr = pd.read_csv(result_dir+'RQ3_qt_LR_explanation_eval_split_rulefit_condition.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])
    result_lr['global_model'] = 'LR'
    
    all_result = pd.concat([result_rf, result_lr])
    all_result['recall'] = all_result['recall']*100
    openstack_result = all_result[all_result['project']=='openstack']
    qt_result = all_result[all_result['project']=='qt']
    
    fig, axs = plt.subplots(1,2, figsize=(8,8))

    axs[0].set_title('Openstack')
    axs[1].set_title('Qt')

    sns.boxplot(data=openstack_result, x='global_model', y='recall', 
                hue='method', ax=axs[0], palette=['darkorange','royalblue']).set(xlabel='', ylabel='Consistency Percentage (%)')
    sns.boxplot(data=qt_result, x='global_model', y='recall', 
                hue='method', ax=axs[1], palette=['darkorange','royalblue']).set(xlabel='', ylabel='')

    
    plt.show()

    openstack_rf = openstack_rf[openstack_rf['method']=='LIME']
    openstack_lr = openstack_lr[openstack_lr['method']=='LIME']
    qt_rf = qt_rf[qt_rf['method']=='LIME']
    qt_lr = qt_lr[qt_lr['method']=='LIME']
    
    get_percent_unique_explanation('openstack','RF', list(openstack_rf['explanation']))
    get_percent_unique_explanation('openstack','LR',list(openstack_lr['explanation']))
    get_percent_unique_explanation('qt','RF',list(qt_rf['explanation']))
    get_percent_unique_explanation('qt','LR',list(qt_lr['explanation']))
    
    fig.savefig(fig_dir+'RQ3.png')

    
def flip_rule(rule):
    rule = re.sub(r'\b=\b',' = ',rule) # for LIME
    found_rule = re.findall('.* <=? [a-zA-Z]+ <=? .*', rule) # for LIME
    ret = ''
    
    # for LIME that has condition like this: 0.53 < nref <= 0.83
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

def get_combined_probs(X_input, global_model, input_guidance, input_SD):

    k_percent = 10
    prob_og = global_model.predict_proba(X_input)[0][1]


    output_df = pd.DataFrame()
    # 1) Revised values must not be negative -> if val < 0 then val = 0
    # 2) Revised values must not be lesser/greater than the actual values for < / > operations
    # E.g., {LOC < 50} => Clean and the actual LOC is 30: 
    # For a 10-percent decrease approach, the revised value according to the threshold is 45
    # In this case (Actual < Revised), we apply the 10-percent decrease approach to the actual value instead of the threshold.
    # Thus, the final revised value is 27 (10-percent decrease from 30)
    # There are 2 approaches:
    # (1) n-percent increase/decrease, e.g., 10-percent
    # (2) an SD_train increase/decrease, e.g., 1SD

    X_revised_sd = X_input.copy()
    for guidance_i in input_guidance.split('&'):
        tmp_g = guidance_i.strip().split(' ')
        if len(tmp_g) != 3:
            continue

        revised_var = tmp_g[0]    
        revised_opr = tmp_g[1] 
        actual_val = X_input[revised_var][0]

        if '<' in revised_opr:
            # < or <=
            revised_from_threshold = float(tmp_g[2]) - input_SD[revised_var]
            revised_from_actual = X_input[revised_var][0] - input_SD[revised_var]
            # the revised value from threshold must not greater than the actual values
            if revised_from_threshold > actual_val:
                actual_revised_value = revised_from_actual
            else:
                actual_revised_value = revised_from_threshold
        else:
            # > or >=
            revised_from_threshold = float(tmp_g[2]) + input_SD[revised_var]
            revised_from_actual = X_input[revised_var][0] + input_SD[revised_var]
            # the revised value from threshold must not less than the actual values
            if revised_from_threshold < actual_val:
                actual_revised_value = revised_from_actual
            else:
                actual_revised_value = revised_from_threshold

        # the revised value must not be negative
        if actual_revised_value < 0:
            actual_revised_value = 0

        X_revised_sd[revised_var] = actual_revised_value

#     prob_revised_percent = global_model.predict_proba(X_revised_percent)[0][1]
    prob_revised_sd = global_model.predict_proba(X_revised_sd)[0][1]
    
    tmp_out = pd.Series(data=[input_guidance, prob_og, prob_revised_sd])
    output_df = output_df.append(tmp_out,ignore_index=True)
    if len(output_df) == 0:
        return []
    output_df.columns = ['guidance', 'probOg', 'probRevisedSD']
    return output_df


def what_if_analysis(proj_name, global_model_name):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)

    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    rq3_explanation_result = pd.DataFrame()

    pyexp_guidance_result_list = []

    x_train_sd = x_train.std()
    
    all_df = pd.DataFrame()

    for i in range(0,len(feature_df)):

        tmp_df = pd.DataFrame()

        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

        exp_obj = pickle.load(open(os.path.join(exp_dir,proj_name,global_model_name,'all_explainer_'+row_index+'.pkl'),'rb'))
        py_exp = exp_obj['pyExplainer']
        lime_exp = exp_obj['LIME']

        # load local models
        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']

        # generate explanations                
        py_exp_the_best_defective_rule_str = get_rule_str_of_rulefit(py_exp_local_model, X_explain)
        lime_the_best_defective_rule_str = lime_exp['rule'].as_list()[0][0]

        # generate guidance
        pyFlip = flip_rule(py_exp_the_best_defective_rule_str)

        tmp_df_i = get_combined_probs(X_explain, global_model, pyFlip, x_train_sd)
        if len(tmp_df_i) > 0:
            tmp_df_i['gtype'] = 'pyFlip'
            tmp_df = tmp_df.append(tmp_df_i)

        tmp_df['commit_id'] = row_index
        tmp_df['model'] = global_model_name
        tmp_df['project'] = proj_name
        all_df = all_df.append(tmp_df)
        
        print('finished {}/{} commits'.format(str(i+1), str(len(feature_df))))

    print('finished what-if of',proj_name)
    all_df.to_csv(result_dir+proj_name+'_'+global_model_name+'_combined_prob_from_guidance.csv',index=False)

def show_what_if_eval_result():
    openstack_rf = pd.read_csv(result_dir+'openstack_RF_combined_prob_from_guidance.csv')
    qt_rf = pd.read_csv(result_dir+'qt_RF_combined_prob_from_guidance.csv')
    result_rf = pd.concat([openstack_rf, qt_rf])

    openstack_lr = pd.read_csv(result_dir+'/openstack_LR_combined_prob_from_guidance.csv')
    qt_lr = pd.read_csv(result_dir+'/qt_LR_combined_prob_from_guidance.csv')
    result_lr = pd.concat([openstack_lr, qt_lr])

    all_result = pd.concat([result_rf, result_lr])

    all_result['predOg'] = all_result['probOg'] >= 0.5
    all_result['predSD'] = all_result['probRevisedSD'] >= 0.5

    pred_og = list(all_result['predOg'])
    pred_sd = list(all_result['predSD'])

    compare_list = []

    for a,b in zip(pred_og, pred_sd):
        compare_list.append(a!=b)

    all_result['isFlip'] = compare_list
    all_result['prob_diff'] = (all_result['probOg']-all_result['probRevisedSD'])*100

    all_result_prob_change = all_result[all_result['prob_diff']>=0]

    openstack_result_prob_change = all_result_prob_change[all_result_prob_change['project']=='openstack']
    qt_result_prob_change = all_result_prob_change[all_result_prob_change['project']=='qt']

    df_list = list(all_result_prob_change.groupby(['project','model']))

    openstack_lr = df_list[0][1]
    openstack_rf = df_list[1][1]
    qt_lr = df_list[2][1]
    qt_rf = df_list[3][1]

    openstack_rf_reverse_percent = np.mean(list(openstack_rf['isFlip']))*100
    openstack_lr_reverse_percent = np.mean(list(openstack_lr['isFlip']))*100
    qt_rf_reverse_percent = np.mean(list(qt_rf['isFlip']))*100
    qt_lr_reverse_percent = np.mean(list(qt_lr['isFlip']))*100

    openstack_reverse_percent_df = pd.DataFrame()
    openstack_reverse_percent_df = openstack_reverse_percent_df.append(pd.Series(['RF', openstack_rf_reverse_percent]), ignore_index=True)
    openstack_reverse_percent_df = openstack_reverse_percent_df.append(pd.Series(['LR', openstack_lr_reverse_percent]), ignore_index=True)
    openstack_reverse_percent_df.columns = ['model','% reverse']

    qt_reverse_percent_df = pd.DataFrame()
    qt_reverse_percent_df = qt_reverse_percent_df.append(pd.Series(['RF', qt_rf_reverse_percent]), ignore_index=True)
    qt_reverse_percent_df = qt_reverse_percent_df.append(pd.Series(['LR', qt_lr_reverse_percent]), ignore_index=True)
    qt_reverse_percent_df.columns = ['model','% reverse']

    # plot probability difference
    plt.figure()

    fig, axs = plt.subplots(1,2)

    fig.suptitle('Probability difference between actual prediction and simulated prediction')
    axs[0].set(ylim=(0, 100))
    axs[1].set(ylim=(0, 100))

    axs[0].set_title('openstack')
    axs[1].set_title('qt')

    sns.boxplot(x='model',y='prob_diff', data=openstack_result_prob_change, ax=axs[0])
    sns.boxplot(x='model',y='prob_diff', data=qt_result_prob_change, ax=axs[1])

    plt.show()

    fig.savefig(fig_dir+'what_if_prob_diff.png')

    # plot percentage of reversed predictions
    plt.figure()

    fig, axs = plt.subplots(1,2)

    fig.suptitle('Percentage of defective commits that their predictions are inversed')

    axs[0].set(ylim=(0, 100))
    axs[1].set(ylim=(0, 100))

    axs[0].set_title('openstack')
    axs[1].set_title('qt')

    sns.barplot(x='model',y='% reverse', data=openstack_reverse_percent_df, ax=axs[0])
    sns.barplot(x='model',y='% reverse', data=qt_reverse_percent_df, ax=axs[1])

    for index, row in openstack_reverse_percent_df.iterrows():
        axs[0].text(row.name,row['% reverse'], round(row['% reverse'],2), color='black', ha="center")

    for index, row in qt_reverse_percent_df.iterrows():
        axs[1].text(row.name,row['% reverse'], round(row['% reverse'],2), color='black', ha="center")

    plt.show()

    fig.savefig(fig_dir+'what_if_percent_reverse_prediction.png')
    
