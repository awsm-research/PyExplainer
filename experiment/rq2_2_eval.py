from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score, r2_score , confusion_matrix, precision_score, recall_score

import pandas as pd
import numpy as np
import seaborn as sns

from my_util import *
from lime.lime.lime_tabular import LimeTabularExplainer

# from pyexplainer.pyexplainer_pyexplainer import PyExplainer
import matplotlib.pyplot as plt

import os, pickle, time, re, sys, operator
from datetime import datetime
from collections import Counter


sys.path.append(os.path.abspath('../pyexplainer'))
from pyexplainer_pyexplainer import *

from IPython.display import display

data_path = './dataset/'
result_dir = './eval_result/'
dump_dataframe_dir = './dump_df/'
pyExp_dir = './pyExplainer_obj/'
other_object_dir = './other_object/'
# proj_name = 'qt' # ['openstack','qt']

def is_in_top_k_global_features(top_k_global_features, the_best_defective_rule_str):
    # remove numeric value
    new_the_best_defective_rule_str = re.sub('\d+','', the_best_defective_rule_str)

    # remove special characters
    new_the_best_defective_rule_str = re.sub('\W+',' ',new_the_best_defective_rule_str)
    splitted_rule = new_the_best_defective_rule_str.split()

    local_feature_count = 0
    
    found_features = set(splitted_rule).intersection(top_k_global_features)
    return list(found_features)

# def eval_rule(rule, X_explain):
#     var_in_rule = re.findall('[a-zA-Z]+',rule)
#     rule = rule.replace('&','and') # just for rulefit
#     rule = re.sub(r'\b=\b','==',rule)
# #             rule = rule.replace('=','==')

#     var_dict = {}

#     for var in var_in_rule:
#         var_dict[var] = float(X_explain[var])

#     eval_result = eval(rule,var_dict)
#     return eval_result

        
def prepare_data_for_testing(proj_name):
    global_model = pickle.load(open(proj_name+'_global_model.pkl','rb'))

    correctly_predict_df = pd.read_csv(dump_dataframe_dir+proj_name+'_correctly_predict_as_defective.csv')
    correctly_predict_df = correctly_predict_df.set_index('commit_id')

    dep = 'defect'
    indep = correctly_predict_df.columns[:-3] # exclude the last 3 columns

    feature_df = correctly_predict_df.loc[:, indep]
    
    return global_model, correctly_predict_df, indep, dep, feature_df

def classification_eval(pred,label):
    f1 = f1_score(label,pred)
    bal_acc = balanced_accuracy_score(label,pred)
    prec = precision_score(label,pred)
    rec = recall_score(label,pred)
    
    return bal_acc, prec, rec, f1

# note: defective commit is correctly predicted as defective commit
def rq2_2_eval(proj_name):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name)
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')
    
    ground_truth = np.ones(len(feature_df)).astype(int)
    
    categorical_features_list = [6]
    class_label = ['clean', 'defect']
    
    lime_explainer = LimeTabularExplainer(x_train.values, categorical_features=categorical_features_list, 
                                      feature_names=indep, class_names=class_label,
                                      random_state=0)

    
    py_exp_pred = []
    lime_pred = []
    
    len_feature_df = len(feature_df)
    
    for i in range(0,len_feature_df):
        X_explain = feature_df.iloc[[i]]

        row_index = str(X_explain.index[0])

#         print('get data done...')
        
        py_exp = pickle.load(open(pyExp_dir+proj_name+'_rulefit_crossoverinterpolation_'+row_index+'.pkl','rb'))
        lime_exp = pickle.load(open(pyExp_dir+proj_name+'_lime_'+row_index+'.pkl','rb'))

#         print('load pickle file done')
        
        py_exp_local_model = py_exp['local_model']
        lime_exp_local_model = lime_exp['local_model']
        
        selected_feature_indices = lime_exp['selected_feature_indeces']
        lime_input = np.ones((1,len(selected_feature_indices)))

#         py_exp_local_prob = py_exp_local_model.predict_proba(X_explain.values)[:,1][0]
        py_exp_local_pred = py_exp_local_model.predict(X_explain.values)[0].astype(int)

#         print('predict py_exp done')
        
        lime_exp_local_prob = lime_exp_local_model.predict(lime_input)[0]
        lime_exp_local_pred = np.round(lime_exp_local_prob).astype(int)

#         print(py_exp_local_pred, lime_exp_local_pred)
#         print('predict lime done')
        
#         print('predict finished')
        
#         py_exp_prob.append(py_exp_local_prob)
        py_exp_pred.append(py_exp_local_pred)
        
#         lime_prob.append(lime_exp_local_prob)
        lime_pred.append(lime_exp_local_pred)
#         print(py_exp_local_prob, py_exp_local_pred, lime_exp_local_prob, lime_exp_local_pred)
#         print(py_exp_pred, lime_pred)
        
#         if py_exp_pred == 1:
#             py_exp_pred_count = py_exp_pred_count +1
#         if lime_pred == 1:
#             lime_pred_count = lime_pred_count + 1

#         del py_exp, lime_exp, py_exp_local_model, lime_exp_local_model, X_explain
    
        print('finished {}/{} instances'.format(str(i+1), len_feature_df))
  
    print('finished all')
    
#     f1 = f1_score(ground_truth,py_exp_pred)
#     print('find f1 done')
#     print(f1)
#     bal_acc = balanced_accuracy_score(ground_truth,py_exp_pred)
#     print('find bal_acc done')
# #     prec = precision_score(ground_truth,py_exp_pred)
# #     print('find prec done')
# #     rec = recall_score(ground_truth,py_exp_pred)
# #     print('find rec done')
    
#     print('result of pyExplainer')
#     print(bal_acc)
    
#     f1 = f1_score(ground_truth,lime_pred)
#     print('find f1 done')
#     print(f1)
#     bal_acc = balanced_accuracy_score(ground_truth,lime_pred)
#     print('find bal_acc done')
# #     prec = precision_score(ground_truth,lime_pred)
# #     print('find prec done')
# #     rec = recall_score(ground_truth,lime_pred)
# #     print('find rec done')
    
#     print('result of LIME')
#     print(bal_acc)
# #     print(py_exp_pred_count, lime_pred_count)
    print('finished RQ555 of',proj_name)
    
    print(ground_truth)
    print(py_exp_pred)
    print(len(ground_truth), len(py_exp_pred))
    tn, fp, fn, tp = confusion_matrix(ground_truth, py_exp_pred).ravel()
    print(tn, fp, fn, tp)

# rq2_2_eval('openstack')
# print('-'*100)
rq2_2_eval('qt')