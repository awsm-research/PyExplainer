#!/usr/bin/env python

import sys, os,  pickle, time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from IPython.display import display
from my_util import *
from lime.lime.lime_tabular import LimeTabularExplainer

sys.path.append(os.path.abspath('../'))
from pyexplainer.pyexplainer_pyexplainer import *

import warnings
warnings.filterwarnings("ignore")

data_path = './dataset/'
result_dir = './eval_result/'
dump_dataframe_dir = './prediction_result/'
exp_dir = './explainer_object/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
if not os.path.exists(dump_dataframe_dir):
    os.makedirs(dump_dataframe_dir)
    
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

def train_global_model(proj_name, x_train,y_train, global_model_name = 'RF'):
    
    smt = SMOTE(k_neighbors=5, random_state=42, n_jobs=24)
    new_x_train, new_y_train = smt.fit_resample(x_train, y_train)
    
    if global_model_name == 'RF':
        global_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=24)
    elif global_model_name == 'LR':
        global_model = LogisticRegression(random_state=0, n_jobs=24)
        
    global_model.fit(new_x_train, new_y_train)
    pickle.dump(global_model, open(proj_name+'_'+global_model_name+'_global_model.pkl','wb'))

def get_correctly_predicted_defective_commit_indices(proj_name, global_model_name, x_test, y_test):

    prediction_df_dir = dump_dataframe_dir+proj_name+'_'+global_model_name+'_prediction_result.csv'
    correctly_predict_df_dir = dump_dataframe_dir+proj_name+'_'+global_model_name+'_correctly_predict_as_defective.csv'
    
    if not os.path.exists(prediction_df_dir) or not os.path.exists(correctly_predict_df_dir):
        global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

        pred = global_model.predict(x_test)
        defective_prob = global_model.predict_proba(x_test)[:,1]

        prediction_df = x_test.copy()
        prediction_df['pred'] = pred
        prediction_df['defective_prob'] = defective_prob
        prediction_df['defect'] = y_test

        correctly_predict_df = prediction_df[(prediction_df['pred']==1) & (prediction_df['defect']==1)]

        prediction_df.to_csv(prediction_df_dir)
        correctly_predict_df.to_csv(correctly_predict_df_dir)
    
    else:
        prediction_df = pd.read_csv(prediction_df_dir)
        correctly_predict_df = pd.read_csv(correctly_predict_df_dir)
        
        prediction_df = prediction_df.set_index('commit_id')
        correctly_predict_df = correctly_predict_df.set_index('commit_id')
        
    return correctly_predict_df.index

def create_explainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, df_indices):
    
    save_dir = os.path.join(exp_dir,proj_name,global_model_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    global_model = pickle.load(open(proj_name+'_'+global_model_name+'_global_model.pkl','rb'))

    indep = x_test.columns
    dep = 'defect'
    class_label = ['clean', 'defect']
    
    # for our apporach
    pyExp = PyExplainer(x_train, y_train, indep, dep, global_model, class_label)

    # for baseline
    # note: 6 is index of 'self' feature
    lime_explainer = LimeTabularExplainer(x_train.values, categorical_features=[6],
                                      feature_names=indep, class_names=class_label, 
                                      random_state=0)

    feature_df = x_test.loc[df_indices]
    test_label = y_test.loc[df_indices]
    
    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]
        y_explain = test_label.iloc[[i]]

        row_index = str(X_explain.index[0])

        pyExp_obj = pyExp.explain(X_explain,
                                   y_explain,
                                   search_function = 'CrossoverInterpolation')
        
        pyExp_obj['commit_id'] = row_index

        # because I don't want to change key name in another evaluation file
        pyExp_obj['local_model'] = pyExp_obj['local_rulefit_model']
        del pyExp_obj['local_rulefit_model']
        
        X_explain = feature_df.iloc[i] # to prevent error in LIME
        exp, synt_inst, synt_inst_for_local_model, selected_feature_indices, local_model = lime_explainer.explain_instance(X_explain, global_model.predict_proba, num_samples=5000)

        lime_obj = {}
        lime_obj['rule'] = exp
        lime_obj['synthetic_instance_for_global_model'] = synt_inst
        lime_obj['synthetic_instance_for_lobal_model'] = synt_inst_for_local_model
        lime_obj['local_model'] = local_model
        lime_obj['selected_feature_indeces'] = selected_feature_indices
        lime_obj['commit_id'] = row_index

        all_explainer = {'pyExplainer':pyExp_obj, 'LIME': lime_obj}
        
        pickle.dump(all_explainer, open(save_dir+'/all_explainer_'+row_index+'.pkl','wb'))
        
        print('finished {}/{} commits'.format(str(i+1), str(len(feature_df))))

def train_global_model_runner(proj_name, global_model_name):
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    train_global_model(proj_name, x_train, y_train,global_model_name)
    print('train {} of {} finished'.format(global_model_name, proj_name))

    
def train_explainer(proj_name, global_model_name):
    x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')

    correctly_predict_indice = get_correctly_predicted_defective_commit_indices(proj_name, global_model_name, x_test, y_test)
    correctly_predict_indice = set(correctly_predict_indice)
    create_explainer(proj_name, global_model_name, x_train, x_test, y_train, y_test, correctly_predict_indice)

proj_name = sys.argv[1]
proj_name = proj_name.lower()
global_model = sys.argv[2]
global_model = global_model.upper()

if proj_name not in ['openstack','qt'] or global_model not in ['RF','LR']:
    print('project name must be "openstack" or "qt".')
    print('global model name must be "RF" or "LR".')
    
else:
    print(proj_name, global_model)
    print('training global model')
    train_global_model_runner(proj_name, global_model)
    print('finished training global model')
    
    print('training explainers')
    train_explainer(proj_name, global_model)
    print('finished training explainers')
