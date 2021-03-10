

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV,LogisticRegressionCV
from functools import reduce
import sklearn
from sklearn.utils import check_random_state
import scipy as sp
from rulefit import RuleCondition, Rule, RuleEnsemble, RuleFit, FriedScale
from sklearn.preprocessing import StandardScaler
import copy
import math
from collections import Counter

class pyExplainer():
    """
    
    TODO
    
    """
    
    def __init__(self,
                X_train,
                y_train,
                indep,
                dep,
                class_label,
                blackbox_model = ''
                ):
        self.X_train = X_train
        self.y_train = y_train
        self.indep = indep
        self.dep = dep
        self.processed_features = []
        self.blackbox_model = blackbox_model
        self.class_label = class_label
        
        
    def __get_min_max_values(self):
        min_values = self.X_train.min()
        max_values = self.X_train.max()
        return {'min_values': min_values,
                'max_values': max_values}
    
        '''
    An approach to generate instance using crossover and interpolation
    '''
    def __generate_instance_crossover_interpolation(self, X_test, y_test, debug = False):

        categorical_vars = []

        X_train_i = self.X_train.copy()
        y_train_i = self.y_train.copy()
        X_test = X_test.copy()
        y_test = y_test.copy()

        X_train_i.reset_index(inplace=True)
        X_test.reset_index(inplace=True)
        X_train_i = X_train_i.loc[:, self.indep]
        y_train_i = y_train_i.reset_index()[[self.dep]]

        X_test = X_test.loc[:, self.indep]
        y_test = y_test.reset_index()[[self.dep]]

        #get the global model predictions for the training set
        target_train = self.blackbox_model.predict(X_train_i)

        # class variables
        ori_dataset = pd.concat([X_train_i.reset_index(drop=True), y_train_i], axis=1)
        colnames = list(ori_dataset)  # Return column names


        # Does feature scaling for continuous data and one hot encoding for categorical data
        scaler = StandardScaler()
        trainset_normalize = X_train_i.copy()
        if debug:
            print(list(X_train_i), "columns")
        cases_normalize = X_test.copy()

        train_objs_num = len(trainset_normalize)
        dataset = pd.concat(objs=[trainset_normalize, cases_normalize], axis=0)
        if debug:
            print(self.indep,"continuous")
            print(type(self.indep))
        dataset[self.indep] = scaler.fit_transform(dataset[self.indep])
        #    dataset = pd.get_dummies(dataset, prefix_sep="__", columns=self.__categorical_vars)
        trainset_normalize = copy.copy(dataset[:train_objs_num])
        cases_normalize = copy.copy(dataset[train_objs_num:])

        temp_df = pd.DataFrame([])

        #make dataframe to store similarities of the trained instances from the explained instance
        dist_df = pd.DataFrame(index=trainset_normalize.index.copy())


        test_num = 0
        width = math.sqrt(len(X_train_i.columns)) * 0.75
        ####################################################### similarity
        for count, case in cases_normalize.iterrows():

            #Calculate the euclidian distance from the instance to be explained
            dist = np.linalg.norm(trainset_normalize.sub(np.array(case)), axis=1)

            #Convert distance to a similarity score
            similarity = np.sqrt(np.exp(-(dist ** 2) / (width ** 2)))

            dist_df['dist'] = similarity
            dist_df['t_target'] = target_train
        #        dist_df
            #get the unique classes of the training set
            unique_classes = dist_df.t_target.unique()
            #Sort similarity scores in to decending order
            dist_df.sort_values(by=['dist'], ascending=False, inplace=True)

        #        dist_df.reset_index(inplace=True)

            # Make a dataframe with top 40 elements in each class
            top_fourty_df = pd.DataFrame([])
            for clz in unique_classes:
                top_fourty_df = top_fourty_df.append(dist_df[dist_df['t_target'] == clz].head(40))

        #        top_fourty_df.reset_index(inplace=True)

            # get the minimum value of the top 40 elements and return the index
            cutoff_similarity = top_fourty_df.nsmallest(1, 'dist', keep='last').index.values.astype(int)[0]

            # Get the location for the given index with the minimum similarity
            min_loc = dist_df.index.get_loc(cutoff_similarity)
            # whole neighbourhood without undersampling the majority class
            train_neigh_sampling_b = dist_df.iloc[0:min_loc + 1]  
            #get the size of neighbourhood for each class
            target_details = train_neigh_sampling_b.groupby(['t_target']).size() 
            if debug:
                    print(target_details, "target_details")
            target_details_df = pd.DataFrame({'target': target_details.index, 'target_count': target_details.values})


            # Get the majority class and undersample 
            final_neighbours_similarity_df = pd.DataFrame([])
            for index, row in target_details_df.iterrows():
                if row["target_count"] > 200:
                    filterd_class_set = train_neigh_sampling_b.loc[
                        train_neigh_sampling_b['t_target'] == row['target']].sample(n=200)
                    final_neighbours_similarity_df = final_neighbours_similarity_df.append(filterd_class_set)
                else:
                    filterd_class_set = train_neigh_sampling_b.loc[train_neigh_sampling_b['t_target'] == row['target']]
                    final_neighbours_similarity_df = final_neighbours_similarity_df.append(filterd_class_set)
                    # print(index,row,"index and row")
            if debug:
                    print(final_neighbours_similarity_df, "final_neighbours_similarity_df")

            #Get the original training set instances which is equal to the index of the selected neighbours
            train_set_neigh = X_train_i[X_train_i.index.isin(final_neighbours_similarity_df.index)]
            if debug:
                    print(train_set_neigh, "train set neigh")
            train_class_neigh = y_test[y_test.index.isin(final_neighbours_similarity_df.index)]
            train_neigh_df = train_set_neigh.join(train_class_neigh)
            class_neigh = train_class_neigh.groupby([self.dep]).size()


            new_con_df = pd.DataFrame([])

            sample_classes_arr = []
            sample_indexes_list = []

            #######Generating 1000 instances using interpolation technique
            for num in range(0, 1000):
                rand_rows = train_set_neigh.sample(2)
                sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                similarity_both = dist_df[dist_df.index.isin(rand_rows.index)]
                sample_classes = train_class_neigh[train_class_neigh.index.isin(rand_rows.index)]
                sample_classes = np.array(sample_classes.to_records().view(type=np.matrix))
                sample_classes_arr.append(sample_classes[0].tolist())

                alpha_n = np.random.uniform(low=0, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                new_ins = x + (y - x) * alpha_n
                new_ins = new_ins.to_frame().T


                # For Categorical Variables

                for cat in categorical_vars:

                    x_df = x.to_frame().T
                    y_df = y.to_frame().T

                    if similarity_both.iloc[0]['dist'] > similarity_both.iloc[1][
                        'dist']:  # Check similarity of x > similarity of y
                        new_ins[cat] = x_df.iloc[0][cat]
                    if similarity_both.iloc[0]['dist'] < similarity_both.iloc[1][
                        'dist']:  # Check similarity of y > similarity of x
                        new_ins[cat] = y_df.iloc[0][cat]
                    else:
                        new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat]])

                new_ins.name = num
                new_con_df = new_con_df.append(new_ins, ignore_index=True)

            #######Generating 1000 instances using cross-over technique
            for num in range(1000, 2000):
                rand_rows = train_set_neigh.sample(3)
                sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                similarity_both = dist_df[dist_df.index.isin(rand_rows.index)]
                sample_classes = train_class_neigh[train_class_neigh.index.isin(rand_rows.index)]
                sample_classes = np.array(sample_classes.to_records().view(type=np.matrix))
                sample_classes_arr.append(sample_classes[0].tolist())

                mu_f = np.random.uniform(low=0.5, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                z = rand_rows.iloc[2]
                new_ins = x + (y - z) * mu_f
                new_ins = new_ins.to_frame().T


                # For Categorical Variables get the value of the closest instance to the explained instance
                for cat in categorical_vars:
                    x_df = x.to_frame().T
                    y_df = y.to_frame().T
                    z_df = z.to_frame().T

                    new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat], z_df.iloc[0][cat]])

                new_ins.name = num
                new_con_df = new_con_df.append(new_ins, ignore_index=True)

            # get the global model predictions of the generated instances and the instances in the neighbourhood
            predict_dataset = train_set_neigh.append(new_con_df, ignore_index=True)
            target = self.blackbox_model.predict(predict_dataset)
            target_df = pd.DataFrame(target)

            neighbor_frequency = Counter(tuple(sorted(entry)) for entry in sample_classes_arr)

            new_df_case = pd.concat([predict_dataset, target_df], axis=1)
            new_df_case = np.round(new_df_case, 2)
            new_df_case.rename(columns={0: y_test.columns[0]}, inplace=True)
            sampled_class_frequency = new_df_case.groupby([self.dep]).size()


            return {'synthetic_data': new_df_case,
                   'sampled_class_frequency': sampled_class_frequency}

    
    '''
    
    This random pertubation approach to generate instances is used by LIME to gerate synthetic instances
    
    '''
    def __generate_instance_random_pertubation(self, X_explain, y_explain, debug = False):

        n_defect_class = 0
        random_seed = 0
        
        data_row = (X_explain.loc[:, self.indep].values)
        num_samples = 1000
        sampling_method = 'gaussian'
        discretizer = None
        sample_around_instance = True
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        scaler.fit(self.X_train.loc[:, self.indep])
        distance_metric='euclidean'
        random_state = check_random_state(random_seed)
        class_names = self.class_label

        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if discretizer is None:
            instance_sample = data_row
            scale = scaler.scale_
            mean = scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]

            if sampling_method == 'gaussian':
                data = random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == 'lhs':
                data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1]*num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)

            if sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = []
            first_row = data_row
        else:
            first_row = discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
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
        if discretizer is not None:
            inverse[1:] = discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row


        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
                scaled_data = (data - scaler.mean_) / scaler.scale_
                distances = sklearn.metrics.pairwise_distances(
                    scaled_data,
                    scaled_data[0].reshape(1, -1),
                    metric=distance_metric
            ).ravel()

        new_df_case = pd.DataFrame(data = scaled_data, columns = self.indep)
        sampled_class_frequency = 0


        n_defect_class = np.sum(self.blackbox_model.predict(new_df_case.loc[:, self.indep]))

        if debug:
            print('Random seed', random_seed, 'nDefective', n_defect_class)

        return {'synthetic_data': new_df_case,
               'sampled_class_frequency': sampled_class_frequency}

    def explain(self, X_explain, y_explain, top_k = 3, max_rules=10, max_iter =10, cv=5, search_function = 'crossoverinterpolation', debug = False):
        
        # Step 1 - Generate synthetic instances
        if search_function == 'crossoverinterpolation':
            
            synthetic_object = self.__generate_instance_crossover_interpolation(
                                          X_explain, 
                                          y_explain, 
                                          debug = debug)
            
        elif search_function == 'randompertubation':
            # This random pertubation approach to generate instances is used by LIME to gerate synthetic instances
            synthetic_object = self.__generate_instance_random_pertubation(
                                          X_explain = X_explain, 
                                          y_explain = y_explain, 
                                          debug = debug)
        else: 
            # TODO
            print('TODO')
        
        # Step 2 - Generate predictions of synthetic instances using the global model
        synthetic_instances = synthetic_object['synthetic_data']
        synthetic_predictions = self.blackbox_model.predict(synthetic_instances.loc[:, self.indep])
        if debug:
            n_defect_class = np.sum(synthetic_predictions)
            print('nDefect=', n_defect_class, 'from', len(synthetic_predictions))
            
        # Step 3 - Build a RuleFit local model with synthetic instances
        indep_index = [list(synthetic_instances.columns).index(i) for i in self.indep]
        local_rulefit_model = RuleFit(rfmode = 'classify',
                              exp_rand_tree_size=False,
                              random_state=0,
                              max_rules=max_rules,
                              max_iter =max_iter,
                              cv = cv,
                              n_jobs = -1)

        local_rulefit_model.fit(synthetic_instances.loc[:, self.indep].values, synthetic_predictions, feature_names=self.indep)
        if debug:
            print('Constructed a RuleFit model')
        # Step 4 Get rules from theRuleFit local model
        rules = local_rulefit_model.get_rules()
        rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
        rules = rules[rules.type == 'rule'].sort_values("importance", ascending=False)
        
        top_k_positive_rules = rules[rules.coef > 0].sort_values("importance", ascending=False).head(top_k)
        top_k_positive_rules['Class'] = self.class_label[1]
        top_k_negative_rules = rules[rules.coef < 0].sort_values("importance", ascending=False).head(top_k)
        top_k_negative_rules['Class'] = self.class_label[0]
        
        return {'synthetic_data': synthetic_instances,
                'synthetic_predictions': synthetic_predictions,
                'X_explain': X_explain,
                'y_explain': y_explain,
                'indep': self.indep,
                'dep': self.dep,
                'top_k_positive_rules': top_k_positive_rules,
                'top_k_negative_rules': top_k_negative_rules}
    
    def get_risk_data(self, X_explain):
        return [{"riskScore": [str(int(round(self.blackbox_model.predict_proba(X_explain)[0][1] * 100, 0))) + '%'],
            "riskPred": [self.class_label[self.blackbox_model.predict(X_explain)[0]]],
            }]

    def parse_top_rules (self, top_k_positive_rules, top_k_negative_rules):
    
        top_variables = []
        top_3_toavoid_rules = []
        top_3_tofollow_rules = []
        for i in range(len(top_k_positive_rules)):
            tmp_rule = (top_k_positive_rules['rule'].iloc[i])
            tmp_rule = tmp_rule.strip()
            tmp_rule = str.split(tmp_rule, '&')
        #    print('tmp_rule:', tmp_rule)
            for j in tmp_rule:
                j=j.strip()
                #print('subrule:', j)
                tmp_sub_rule = str.split(j, ' ')
        #        print(tmp_sub_rule)
                tmp_variable = tmp_sub_rule[0]
                tmp_condition_variable = tmp_sub_rule[1]
                tmp_value = tmp_sub_rule[2]

                if tmp_variable not in top_variables:
                    top_variables.append(tmp_variable)
                    top_3_toavoid_rules.append({
                        'variable': tmp_variable,
                        'lessthan': tmp_condition_variable[0] == '<',
                        'value': tmp_value
                    })

                #print(tmp_variable, tmp_condition_variable, tmp_value)
                if len(top_3_toavoid_rules) == 3:
                    break
            if len(top_3_toavoid_rules) == 3:
                    break

        for i in range(len(top_k_negative_rules)):

            tmp_rule = (top_k_negative_rules['rule'].iloc[i])
            tmp_rule = tmp_rule.strip()
            tmp_rule = str.split(tmp_rule, '&')
        #    print('tmp_rule:', tmp_rule)
            for j in tmp_rule:
                j=j.strip()
                #print('subrule:', j)
                tmp_sub_rule = str.split(j, ' ')
        #        print(tmp_sub_rule)
                tmp_variable = tmp_sub_rule[0]
                tmp_condition_variable = tmp_sub_rule[1]
                tmp_value = tmp_sub_rule[2]

                if tmp_variable not in top_variables:
                    top_variables.append(tmp_variable)
                    top_3_tofollow_rules.append({
                        'variable': tmp_variable,
                        'lessthan': tmp_condition_variable[0] == '<',
                        'value': tmp_value
                    })

                #print(tmp_variable, tmp_condition_variable, tmp_value)
                if len(top_3_tofollow_rules) == 3:
                    break
            if len(top_3_tofollow_rules) == 3:
                    break
        return {'top_tofollow_rules': top_3_tofollow_rules,
               'top_toavoid_rules': top_3_toavoid_rules}
    
    
    def get_bullet_data(self, parsed_rule_object, X_explain):


        min_max_values = self.__get_min_max_values()
        # Version 01 - only visualise for what to follow (Rules => Clean)
        bullet_data = []

        
        for i in range(len(parsed_rule_object['top_tofollow_rules'])):
            #         {'variable': 'MAJOR_COMMIT', 'lessthan': True, 'value': '1.550000011920929'}
            tmp_rule = parsed_rule_object['top_tofollow_rules'][i]
            tmp_min = int((min_max_values['min_values'][tmp_rule['variable']]))
            tmp_max = int(round(min_max_values['max_values'][tmp_rule['variable']]))
            tmp_interval = (tmp_max - tmp_min) / 10.0
            tmp_threshold_value = round(float(tmp_rule['value']), 2)
            tmp_actual_value = round(X_explain[tmp_rule['variable']][0], 2)

            plot_min = int(round(max(tmp_min, tmp_threshold_value - tmp_interval),0))*1.0
            plot_max = int(round(min(tmp_max, tmp_threshold_value + tmp_interval), 0))*1.0
            diff_plot_max_min = plot_max - plot_min
            
            print('Min', tmp_min, 'Max', tmp_max, 'threshold', tmp_threshold_value, 'Actual', tmp_actual_value, 'Plot_min', plot_min, 'Plot_max', plot_max)

            tmp_subtitle_text = 'Actual = '+str(tmp_actual_value)
            tmp_ticks = [plot_min, plot_max]
            tmp_step = [1]
            tmp_startpoints = [0, round((tmp_threshold_value - plot_min)/diff_plot_max_min * 760, 0)]
            tmp_widths = [round((tmp_threshold_value - plot_min)/diff_plot_max_min * 760, 0), round((plot_max - tmp_threshold_value)/diff_plot_max_min * 760, 0)]
            tmp_markers = [tmp_actual_value]
            
            if tmp_rule['lessthan']:

                # lessthan == TRUE:
                # The rule suggest to decrease the values to less than a certain threshold
                tmp_title_text = '#'+str(i+1)+' Decrease the values of '+tmp_rule['variable']+' to less than '+ str(tmp_actual_value)           
                tmp_colors = ["#a6d96a", "#d7191c"]

            else:

                # lessthan == FALSE:
                # The rule suggest to increase the values to more than a certain threshold
                tmp_title_text = '#'+str(i+1)+' Increase the values of '+tmp_rule['variable']+' to more than '+ str(tmp_actual_value)
                tmp_colors = ["#d7191c", "#a6d96a"]

            bullet_data.append({
              "title": tmp_title_text,
              "subtitle": tmp_subtitle_text,
              "ticks": tmp_ticks,
              "step": tmp_step,
              "startPoints": tmp_startpoints,
              "widths": tmp_widths,
              "colors": tmp_colors,
              "markers": tmp_markers,
              "tickFormat": tmp_ticks
            })
            
        return (bullet_data)
            
