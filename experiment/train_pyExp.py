import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

import os, pickle, time, sys

sys.path.append(os.path.abspath('../pyexplainer'))
from pyexplainer_pyexplainer import *

import matplotlib.pyplot as plt


from datetime import datetime

from my_util import *

parser = argparse.ArgumentParser()
parser.add_argument('-proj_name', type=str, default='openstack', help='project name (openstack or qt)')
parser.add_argument('-local_model_type',type=str, default='LRR', help='local model type (rulefit or LRR)')

args = parser.parse_args()

proj_name = args.proj_name
local_model_type = args.local_model_type

# print(proj_name, local_model_type)

data_path = './dataset/'
result_dir = './eval_result/'
dump_dataframe_dir = './dump_df/'
pyExp_dir = './pyExplainer_obj/'
other_object_dir = './other_object/'

def train_global_model(x_train,y_train):
    global_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=24)
    global_model.fit(x_train, y_train)

    pickle.dump(global_model, open(proj_name+'_global_model.pkl','wb'))
    print('train global model finished')
    
def create_pyExplainer_obj(search_function, feature_df, test_label, explainer='LRR'):
    
    if search_function.lower() not in ['lime','crossoverinterpolation']:
        print('the search function must be "lime" or "crossoverinterpolation"')
        return
    
    problem_index = []
    time_spent = []
    
    for i in range(0,len(feature_df)):
        X_explain = feature_df.iloc[[i]]
        y_explain = test_label.iloc[[i]]

        row_index = str(X_explain.index[0])

        start = time.time()
        try:
            if search_function=='CrossoverInterpolation':
                # the returned object is dictionary
                pyExp_obj = pyExp.explain(X_explain,
                                           y_explain,
                                           search_function = search_function, 
                                           top_k = 15,
                                           max_rules=2000, 
                                           max_iter = None, 
                                           cv=5,
                                           debug = False)
    #             synt_pred = pyExp_obj['synthetic_predictions']
                pyExp_obj['commit_id'] = row_index
        
                # because I don't want to change key name in another evaluation file
                pyExp_obj['local_model'] = pyExp_obj['local_rulefit_model']
                del pyExp_obj['local_rulefit_model']
    #             print('{}: found {} defect from total {}'.format(row_index, str(np.sum(synt_pred)), 
    #                                                          str(len(synt_pred))))
                pickle.dump(pyExp_obj, open(pyExp_dir+proj_name+'_'+explainer+'_'+search_function.lower()+'_'+row_index+'_2000_instances.pkl','wb'))
        
            else:
                X_explain = feature_df.iloc[i] # to prevent error in LIME
                exp, synt_inst, synt_inst_for_local_model, selected_feature_indices, local_model = lime_explainer.explain_instance(X_explain, global_model.predict_proba)

                lime_obj = {}
                lime_obj['rule'] = exp
                lime_obj['synthetic_instance_for_global_model'] = synt_inst
                lime_obj['synthetic_instance_for_lobal_model'] = synt_inst_for_local_model
                lime_obj['local_model'] = local_model
                lime_obj['selected_feature_indeces'] = selected_feature_indices
                lime_obj['commit_id'] = row_index
                pickle.dump(lime_obj, open(pyExp_dir+proj_name+'_lime_'+row_index+'.pkl','wb'))
                
            print('finished',row_index)
#             print(row_index)
#             print('just one rulefit is enough')
#             break
            
        except Exception as e:
            problem_index.append(row_index)
            print('-'*100)
            print(e)
#             print('found total {} problematic commit'.format(str(len(problem_index))))
            print('-'*100)
            
#         break

        end = time.time()

        time_spent.append(str(end-start))
#     print(row_index)
#     break
    
    print('from total {} commits, there are {} problematic commits'.format(len(feature_df),len(problem_index)))
    return time_spent, problem_index

x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')
col = list(x_test.columns)

smt = SMOTE(k_neighbors=5, random_state=42, n_jobs=24)
# enn = EditedNearestNeighbours(n_neighbors=5, n_jobs=24)
# smt_tmk = SMOTETomek(smote = smt, random_state=0)
# smt_enn = SMOTEENN(smote=smt, enn=enn, random_state=0)

new_x_train, new_y_train = smt.fit_resample(x_train, y_train)

train_black_box = True

if train_black_box:
    train_global_model(new_x_train, new_y_train)

global_model = pickle.load(open(proj_name+'_global_model.pkl','rb'))

pred = global_model.predict(x_test)
defective_prob = global_model.predict_proba(x_test)[:,1]

prediction_df = x_test.copy()
prediction_df['pred'] = pred
prediction_df['defective_prob'] = defective_prob
prediction_df['defect'] = y_test

correctly_predict_df = prediction_df[(prediction_df['pred']==1) & (prediction_df['defect']==1)]

prediction_df.to_csv(dump_dataframe_dir+proj_name+'_prediction_result.csv')
correctly_predict_df.to_csv(dump_dataframe_dir+proj_name+'_correctly_predict_as_defective.csv')

class_label = ['clean', 'defect']
dep = 'defect'
indep = correctly_predict_df.columns[:-3]

pyExp = PyExplainer(x_train, y_train, indep, dep, global_model, class_label)

feature_df = correctly_predict_df.loc[:, indep]
test_label = correctly_predict_df.loc[:, dep]

# time_spent_rand, problem_index_rand = create_pyExplainer_obj('randompertubation', feature_df, test_label,local_model_type)
# pickle.dump(time_spent_rand, open(other_object_dir+proj_name+'_train_time_'+local_model_type+'_randompertubation.pkl','wb'))
# pickle.dump(problem_index_rand, open(other_object_dir+proj_name+'_problem_index_'+local_model_type+'_randompertubation.pkl','wb'))

time_spent_ci, problem_index_ci = create_pyExplainer_obj('CrossoverInterpolation', feature_df, test_label,local_model_type)
pickle.dump(time_spent_ci, open(other_object_dir+proj_name+'_train_time_'+local_model_type+'_crossoverinterpolation.pkl','wb'))
pickle.dump(problem_index_ci, open(other_object_dir+proj_name+'_problem_index_'+local_model_type+'_crossoverinterpolation.pkl','wb'))




