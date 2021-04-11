import pandas as pd
from datetime import datetime

data_path = './dataset/'

def load_change_metrics_df(cur_proj):
    if cur_proj == 'qt':
        start = 1308350292
        end = 1395090476
    elif cur_proj == 'openstack':
        start = 1322599384
        end = 1393590700
    change_metrics = pd.read_csv(data_path+cur_proj+'_metrics.csv')
    
    change_metrics = change_metrics[(change_metrics['author_date'] >= start) & 
                                    (change_metrics['author_date'] <= end)]
    
    change_metrics['self'] = [1 if s is True else 0 for s in change_metrics['self']]
    change_metrics['defect'] = change_metrics['bugcount'] > 0
#     change_metrics['new_date'] = change_metrics['author_date'].apply(lambda x: datetime.fromtimestamp(x).strftime('%m-%d-%Y'))
    change_metrics['new_date'] = change_metrics['author_date'].apply(lambda x: datetime.fromtimestamp(x))
    
    change_metrics = change_metrics.sort_values(by='new_date')
    change_metrics['new_date'] = change_metrics['new_date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    
    change_metrics['rtime'] = (change_metrics['rtime']/3600)/24
    change_metrics['age'] = (change_metrics['age']/3600)/24
    change_metrics = change_metrics.reset_index()
    change_metrics = change_metrics.set_index('commit_id')
    
    bug_label = change_metrics['defect']
    
    # use commit_id as index (so don't remove it)
    change_metrics = change_metrics.drop(['author_date', 'new_date', 'bugcount','fixcount','revd','tcmt','oexp','orexp','osexp','osawr','defect']
                                         ,axis=1)
    change_metrics = change_metrics.fillna(value=0)
    
    
    return change_metrics, bug_label

def split_train_test_data(feature_df, label, percent_split = 70):
    _p_percent_len = int(len(feature_df)*(percent_split/100))
    x_train = feature_df.iloc[:_p_percent_len]
    y_train = label.iloc[:_p_percent_len]
    
    x_test = feature_df.iloc[_p_percent_len:]
    y_test = label.iloc[_p_percent_len:]
    
    return x_train, x_test, y_train, y_test

def prepare_data(proj_name, mode = 'all'):
    if mode not in ['train','test','all']:
        print('this function accepts "train","test","all" mode only')
        return
    
    change_metrics, bug_label = load_change_metrics_df(proj_name) 
    
    with open(data_path+proj_name+'_non_correlated_metrics.txt','r') as f:
        metrics = f.read()
    
    metrics_list = metrics.split('\n')
    
    non_correlated_change_metrics = change_metrics[metrics_list]

    x_train, x_test, y_train, y_test = split_train_test_data(non_correlated_change_metrics, bug_label, percent_split = 70)
    
    if mode == 'train':
        return x_train,y_train
    elif mode == 'test':
        return x_test, y_test
    elif mode == 'all':
        return x_train, x_test, y_train, y_test
    
def prepare_data_all_metrics(proj_name, mode = 'all'):
    if mode not in ['train','test','all']:
        print('this function accepts "train","test","all" mode only')
        return
    
    change_metrics, bug_label = load_change_metrics_df(proj_name) 

    x_train, x_test, y_train, y_test = split_train_test_data(change_metrics, bug_label, percent_split = 70)
    
    if mode == 'train':
        return x_train
    elif mode == 'test':
        return x_test
    elif mode == 'all':
        return x_train, x_test