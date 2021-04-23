import sys, os,  pickle, time

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from my_util import *
from evaluation_function import *
from lime.lime.lime_tabular import LimeTabularExplainer

# sys.path.append(os.path.abspath('../'))
from pyexplainer.pyexplainer_pyexplainer import *

from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

data_path = './dataset/'
result_dir = './new_eval_result/'
dump_dataframe_dir = './prediction_result/'
exp_dir = './explainer_obj_20_4_2021/'

def rq3_eval(proj_name, global_model_name):
    global_model, correctly_predict_df, indep, dep, feature_df = prepare_data_for_testing(proj_name, global_model_name)
    x_test, y_test = prepare_data(proj_name, mode = 'test')

    rq3_explanation_result = pd.DataFrame()
    
    pyexp_guidance_result_list = []
    lime_guidance_result_df = pd.DataFrame()
    
    for i in range(0,len(feature_df)):
#     for i in range(0,5):
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
    rq3_explanation_result.to_csv(result_dir+'RQ3_'+proj_name+'_'+global_model_name+'_guidance_eval_split_rulefit_condition.csv',
                                  index=False)
    
    print('finished')

rq3_eval(sys.argv[1], sys.argv[2])