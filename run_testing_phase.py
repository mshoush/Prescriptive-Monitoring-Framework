import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from tqdm import tqdm
import threading
import random
import time
import gc
from hyperopt.fmin import generate_trials_to_calculate

# show all columns in pandas
pd.set_option('display.max_columns', None)




from hyperopt import hp



def create_search_space_filter(max_ub_te, max_ub_iw, rule_name="predictive_conformal"):
    space_dict = {

    # pridictive    
    "predictive": {"neg_proba": [0.5, 0.9, 0.1]
                    },

    # predictive_conformal
    "predictive_conformal": {"neg_proba": [0.5, 0.9, 0.1],
                             "confidence_level": [0.1, 0.9, 0.1]
                             },  

    # causal
    "causal": {"te": [0.0, 1.0, 0.1]
                },

    # causal_conformal
    "causal_conformal": {"te": [0.0, 1.0, 0.1],
                         "causal_uncer": [0.0, max_ub_te, 0.1],
                         "confidence_level": [0.1, 0.9, 0.1]
                         }, 

    # urgency_conformal
    "urgency_conformal": {"iw": [0.0, max_ub_iw, 0.1],
                        "confidence_level": [0.1, 0.9, 0.1]
                        }, 

    # combined
    # predictive_conformal_urgency_conformal
    "predictive_conformal_urgency_conformal": {"neg_proba": [0.5, 0.9, 0.1],
                                                "iw": [0.0, max_ub_iw, 0.1],
                                                "confidence_level": [0.1, 0.9, 0.1]
                                                },

    # causal_conformal_urgency_conformal
    "causal_conformal_urgency_conformal": {"te": [0.0, 1.0, 0.1],
                                         "iw": [0.0, max_ub_iw, 0.1],
                                         "causal_uncer": [0.0, max_ub_te, 0.1],
                                         "confidence_level": [0.1, 0.9, 0.1]
                                        },

    # predictive_conformal_causal_conformal
    "predictive_conformal_causal_conformal": {"te": [0.0, 1.0, 0.1],
                                            "neg_proba": [0.5, 0.9, 0.1],
                                            "causal_uncer": [0.0, max_ub_te, 0.1],
                                            "confidence_level": [0.1, 0.9, 0.1]
                                            },


    # all
    # predictive_conformal_causal_conformal_urgency_conformal
    "predictive_conformal_causal_conformal_urgency_conformal": {"te": [0.0, 1.0, 0.1],
                                                    "neg_proba": [0.5, 0.9, 0.1],
                                                    "causal_uncer": [0.0, max_ub_te, 0.1],
                                                    "iw": [0.0, max_ub_iw, 0.1],
                                                    "confidence_level": [0.1, 0.9, 0.1]
                                                    },       
                        }

    search_space = {}
    params = space_dict.get(rule_name, {})
    

    for param_name, param_range in params.items():
        param_space = hp.uniform(f'{param_name}', param_range[0], param_range[1])
        search_space[param_name] = param_space
    return search_space, space_dict

# Example usage:
search_space, space_dict = create_search_space_filter(3, 3, rule_name="predictive")
rules = list(space_dict.keys())


import os
import random
import numpy as np

def create_results_directory(log_name, iteration, result_dir, t_dur=1, t_dist="fixed"):
    if t_dist == "normal":
        treatment_duration = int(random.uniform(1, t_dur))
        folder = f"{result_dir}/results_normal_{iteration}/{log_name}/"
    elif t_dist == "fixed":
        treatment_duration = t_dur
        folder = f"{result_dir}/results_fixed_{iteration}/{log_name}/"
    else:
        treatment_duration = int(np.random.exponential(t_dur, size=1))
        folder = f"{result_dir}/results_exp_{iteration}/{log_name}/"

    results_dir = f"{folder}"

    if not os.path.exists(os.path.join(results_dir)):
        os.makedirs(os.path.join(results_dir))

    return results_dir, treatment_duration, t_dist



def sum_gain_treated_cases(treated_cases):
    sum_gain = {
        'TN': 0,  # True Negative
        'FP': 0,  # False Positive
        'TP': 0,  # True Positive
        'FN': 0,  # False Negative
        'gain': 0
    }

    for case in treated_cases.values():
        actual = case[3]  # Assuming actual outcomes are in the 4th column
        predicted = case[4]  # Assuming predicted outcomes are in the 5th column
        gain = case[-1]  # Assuming gain values are in the second-to-last column

        if actual == 0 and predicted == 0:  # True Negative
            sum_gain['TN'] += gain
        elif actual == 0 and predicted == 1:  # False Positive
            sum_gain['FP'] += gain
        elif actual == 1 and predicted == 1:  # True Positive
            sum_gain['TP'] += gain
        elif actual == 1 and predicted == 0:  # False Negative
            sum_gain['FN'] += gain
        
        sum_gain['gain'] += gain

    return sum_gain

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

def auc(tp, fp, fn, tn):
    return tp / (tp + fn) + tn / (tn + fp) if (tp + fn) != 0 and (tn + fp) != 0 else 0


def get_alpha_col(confidence_level):
        alpha_value = np.round((1 - confidence_level), 1)
        alpha_col = "alpha_" + str(alpha_value)
        return alpha_value, alpha_col

def allocate_res(s_res, dist, nr_res, treatment_duration):
    t = threading.Thread(daemon=True, target=block_and_release_res, args=(s_res, dist, nr_res, treatment_duration))
    t.start()

def block_and_release_res(s_res, dist, nr_res, treatment_duration):
    time.sleep(treatment_duration)
    nr_res.append(s_res)



class FilterCases:

    def __init__(self):
        pass
    
    def get_alpha_col(self, confidence_level):
        alpha_value = np.round((1 - confidence_level), 1)
        alpha_col = "alpha_" + str(alpha_value)
        return alpha_value, alpha_col


    def filter_cases(self, row, iw_thre, te_thre, neg_proba_thre, causal_uncer_thre, proba_uncer_thre, confidence_level, rule="predictive"):
        """
        Filter cases based on rule
        """

        # get alpha value and alpha col    
        self.confidence = confidence_level
        alpha_value, alpha_col = self.get_alpha_col(self.confidence)

        # determine values for unique params: params: ['iw', 'proba_uncer', 'te', 'neg_proba', 'causal_uncer', 'proba_conformal']
        # iw: depends on confidence_level
        lb_iw = row[f'lower_bound_{alpha_value}']
        ub_iw = row[f'upper_bound_{alpha_value}']
        iw = (lb_iw + ub_iw) / 2

        # proba_uncer
        proba_uncer = row['total_uncer']

        # te
        te = row['CATE']

        # neg_proba
        neg_proba = row['predicted_proba_0']

        # causal_uncer: depends on confidence_level
        lb_te = row[f'lower_counterfactual_1_alpha_{alpha_value}']
        ub_te = row[f'upper_counterfactual_1_alpha_{alpha_value}']
        causal_uncer = (lb_te + ub_te) / 2

        # proba_conformal: depends on confidence_level
        proba_conformal = row[alpha_col]

        
        # filter cases
        if rule == "predictive":
            #print("Predicitve neg_proba: ", neg_proba, neg_proba_thre)
            if neg_proba > neg_proba_thre:
                return row 
            else:
                return None

        elif rule == "predictive_conformal":
            if neg_proba > neg_proba_thre and proba_conformal == 0:
                return row 
            else:
                return None


        elif rule == "causal":
            if te > te_thre:
                return row 
            else:
                return None


        elif rule == "causal_conformal":
            if te > te_thre and causal_uncer > causal_uncer_thre:
                return row
            else:
                return None


        elif rule == "urgency_conformal":
            if iw < iw_thre:
                return row
            else:
                return None

        
        elif rule == "predictive_conformal_urgency_conformal":
            if neg_proba > neg_proba_thre and proba_conformal == 0 and iw < iw_thre:
                return row
            else:
                return None

        elif rule == "predictive_conformal_causal_conformal":
            if neg_proba > neg_proba_thre and proba_conformal == 0 and te > te_thre and causal_uncer > causal_uncer_thre:
                return row
            else:
                return None

        elif rule == "causal_conformal_urgency_conformal":
            if te > te_thre and causal_uncer > causal_uncer_thre and iw < iw_thre:
                return row
            else:
                return None

        elif rule == "predictive_conformal_causal_conformal_urgency_conformal":
            if  neg_proba > neg_proba_thre and proba_conformal == 0 and te > te_thre and causal_uncer > causal_uncer_thre and iw < iw_thre:
                return row
            else:
                return None

        else:
            print("No valid rule provided")





# define run rule function
def run_rule(data_dict, resources, rule_name, treatment_duration, iw_thre, te_thre, neg_proba_thre, causal_uncer_thre, proba_uncer_thre,  
                weight_need, weight_effect, weight_urgency, weight_uncertainty,
                confidence_level=0.9):
    """
    Run rule
    """
    # sleep for 10 seconds to simulate the runtime of the rule
    time.sleep(15)

    # get alpha value and alpha col
    filter_cases = FilterCases()
    confidence_level = np.round(confidence_level, 1)
    alpha_value, alpha_col = filter_cases.get_alpha_col(confidence_level)

    # availble resources
    availble_resources = list(range(1, resources + 1, 1))
    print(f"availble_resources: {availble_resources}")

    candidate_cases = {}
    treated_cases = {}

    treatmnet_cost =  25 
    outcom_gain = 50


    for row in data_dict:
        if row['case_id'] not in treated_cases.keys():
            if filter_obj.filter_cases(row, iw_thre, te_thre, neg_proba_thre, causal_uncer_thre, proba_uncer_thre, confidence_level, rule=rule_name):
                alpha_value, alpha_col = get_alpha_col(confidence_level)

                # determine values for unique params: params: ['iw', 'proba_uncer', 'te', 'neg_proba', 'causal_uncer', 'proba_conformal']
                # iw: depends on confidence_level
                lb_iw = row[f'lower_bound_{alpha_value}']
                ub_iw = row[f'upper_bound_{alpha_value}']
                iw = (lb_iw + ub_iw) / 2

                # proba_uncer
                proba_uncer = row['total_uncer']

                # te
                te = row['CATE']

                # neg_proba
                neg_proba = row['predicted_proba_0']

                # causal_uncer: depends on confidence_level
                lb_te = row[f'lower_counterfactual_1_alpha_{alpha_value}']
                ub_te = row[f'upper_counterfactual_1_alpha_{alpha_value}']
                causal_uncer = (lb_te + ub_te) / 2

                # proba_conformal: depends on confidence_level
                proba_conformal = row[alpha_col]

                case_id = row['case_id']
                activity = row['activity']
                timestamp = row['timestamp']
                actual = row['actual']
                predicted = row['predicted']

                y0 = row['y0']#row['y0']
                y1 = row['y1']
                t = row['t']
                y = y1 if t==1 else y0
                outcome = 0 if y0 >= 0.5 else 1 if y1 >= 0.5 else 0
                outcomey = 0 if y < 0.5 else 1


                candidate_cases[case_id] = [case_id, activity, timestamp, actual, predicted,
                                            neg_proba, proba_uncer, te, iw, causal_uncer, proba_conformal, y0, y1, t, y, outcome, outcomey            
                                            ]

                best_case_key = rank_candidate_cases(candidate_cases, rule_name, weight_need, weight_effect, weight_urgency, weight_uncertainty,)

                if best_case_key and availble_resources:
                    selceted_res = availble_resources[0]
                    availble_resources.remove(availble_resources[0])

                    y00 = candidate_cases[best_case_key][-6]
                    y11 = candidate_cases[best_case_key][-5]
                    tt = candidate_cases[best_case_key][-4]
                    yy = candidate_cases[best_case_key][-3]
                    outcome2 = candidate_cases[best_case_key][-2]
                    outcomeyy = candidate_cases[best_case_key][-1]

                    netGain = outcomeyy * (outcom_gain - treatmnet_cost)

                    treated_cases[best_case_key] = candidate_cases[best_case_key]

                    treated_cases[best_case_key].append(netGain1)
                    treated_cases[best_case_key].append(netGain2)
                    treated_cases[best_case_key].append(netGain3)
                    treated_cases[best_case_key].append(netGainn)
                    treated_cases[best_case_key].append(netGain)

                    del candidate_cases[best_case_key]
                    allocate_res(selceted_res, t_dist, availble_resources, treatment_duration)

    print(f"\n{rule_name}: {len(treated_cases)}")
    result = sum_gain_treated_cases(treated_cases)
    print(f"Sum of Gain for Treated Cases with rule: {rule_name} is: {result['gain']}")
    
    return treated_cases



def rank_candidate_cases(candidate_cases, rule_name, weight_need=1, weight_effect=1, weight_urgency=1, weight_uncertainty=1):
    """
    Rank candidate cases based on modified criteria with weights.
    """
    if candidate_cases:
        best_case_key = None
        best_case_score = float('-inf')



        for case_key, case in candidate_cases.items():
            need_proba = case[5]  # Assuming need probability is in the 6th column
            effect = case[9] # Assuming effect is in the 10th column
            urgency = case[8]
            uncertainty = case[6] + case[10]  # Assuming uncertainty is in the 7th and 11th columns


            score = (weight_need * need_proba) + (weight_effect * effect) + (weight_urgency * urgency) - (weight_uncertainty * uncertainty)

            if score > best_case_score:
                best_case_score = score
                best_case_key = case_key

        return best_case_key
    else:
        return None



from sys import argv

log_name = argv[1]
result_dir = argv[2]
resourcesAll = int(argv[3])
iteration = argv[4]

results_directory, treatment_duration, t_dist = create_results_directory(log_name, iteration, result_dir=result_dir, t_dur=1, t_dist="fixed")

valid_all = pd.read_csv(f"./results/valid_{log_name}_all.csv", sep=";")
test_all = pd.read_csv(f"./results/test_{log_name}_all.csv", sep=";")


# df into dict
valid_all_dict = valid_all.to_dict("records")
test_all_dict = test_all.to_dict("records")

del test_all


from hyperopt import fmin, hp, tpe, Trials
import numpy as np  
from functools import partial



def objective(params, valid_all_dict, resources,rule_name,):
    rule_treated_cases_dict = {}

    # Define the default parameters
    best_params_default = {
        'neg_proba': 0.5, 'proba_conformal': 0.0, 'proba_uncer': 0.5, 'te': 0.0,
        'causal_uncer': 0.0, 'iw': 10, 'confidence_level': 0.9,
        'weight_need': 1, 'weight_effect': 1, 'weight_urgency': 1, 'weight_uncertainty': 1,
    }
    
    params = {**best_params_default, **params}
    params['rule_name'] = rule_name
    print(f"Params Objective: {params}")


    weights = {
        'weight_need': params['weight_need'],
        'weight_effect': params['weight_effect'],
        'weight_urgency': params['weight_urgency'],
        'weight_uncertainty': params['weight_uncertainty'],
    }

    filter_params = {
        'neg_proba': params['neg_proba'],
        'proba_conformal': params['proba_conformal'],
        'proba_uncer': params['proba_uncer'],
        'te': params['te'],
        'causal_uncer': params['causal_uncer'],
        'iw': params['iw'],
        'confidence_level': params['confidence_level'],
    }



    # Assuming you have valid_data_dict, resources, and treatment_duration defined
    rule_treated_cases_dict[params['rule_name']] = run_rule(
        valid_all_dict, resources,rule_name, treatment_duration,
        filter_params.get('iw', best_params_default['iw']),
        filter_params.get('te', best_params_default['te']),
        filter_params.get('neg_proba', best_params_default['neg_proba']),
        filter_params.get('causal_uncer', best_params_default['causal_uncer']),
        filter_params.get('proba_uncer', best_params_default['proba_uncer']),
        weights['weight_need'], weights['weight_effect'], weights['weight_urgency'], weights['weight_uncertainty'],
        filter_params.get('confidence_level', best_params_default['confidence_level'])
    )

    result = sum_gain_treated_cases(rule_treated_cases_dict[params['rule_name']])

    # Minimize the negative of netGain
    return -result['gain']

from datetime import datetime





from datetime import datetime
from functools import partial
from hyperopt import fmin, hp, tpe, Trials
import numpy as np
import pickle

def run_testing_phase(resources_list, filter_obj, hyper_opt_argv, best_params_default):


    results_dict_hyperopt = {}
    results_dict_no_hyperopt = {}
    time_taken_dict = {}
    best_trial_dict = {}
    best_params_dict = {}

    confidence_level = best_params_default['confidence_level']
    alpha_value = np.round((1 - confidence_level), 1)

    time_taken_per_rule_dict = {}  # Add this dictionary to store time taken per rule
    init_vals = [best_params_default]

    for resources in resources_list:
        print("\n==================================resources:", resources, "==================================\n")

        for rule_name in rules:
            best_params = {}

            if hyper_opt_argv:
                print(f"hyperopt_enabled: {hyper_opt_argv}")

                space_w = {
                    'weight_need': hp.choice('weight_need', [0.5, 1.0, 2.0]),
                    'weight_effect': hp.choice('weight_effect', [0.5, 1.0, 2.0]),
                    'weight_urgency': hp.choice('weight_urgency', [0.5, 1.0, 2.0]),
                    'weight_uncertainty': hp.choice('weight_uncertainty', [0.5, 1.0, 2.0]),
                }

                space_p, space_dict = create_search_space_filter(max_ub_te, max_ub_iw, rule_name=rule_name)
                combined_space = {**space_w, **space_p}


                #trials = Trials()
                trials = generate_trials_to_calculate(init_vals)
                start_time = datetime.now()
                partial_objective = partial(objective, rule_name=rule_name, valid_all_dict=valid_all_dict, resources=resources)

                best_params = fmin(fn=partial_objective, space=combined_space, algo=tpe.suggest, max_evals=50, trials=trials)
                best_params_dict.setdefault(resources, {})[rule_name] = best_params
                best_params = {**best_params_default, **best_params}
                end_time = datetime.now()
                time_taken = end_time - start_time

                time_taken_dict.setdefault(resources, {})[rule_name] = time_taken.total_seconds()
                best_trial_dict.setdefault(resources, {})[rule_name] = trials.best_trial

            else:
                best_params = best_params_default

            print(f"Running rule: {rule_name}")
            start_time_per_rule = datetime.now()
            rule_treated_cases_dict[rule_name] = run_rule(
                test_all_dict, resources, rule_name, treatment_duration,
                best_params.get('iw', best_params_default['iw']),
                best_params.get('te', best_params_default['te']),
                best_params.get('neg_proba', best_params_default['neg_proba']),
                best_params.get('causal_uncer', best_params_default['causal_uncer']),
                best_params.get('proba_uncer', best_params_default['proba_uncer']),
                best_params.get('weight_need', best_params_default['weight_need']),
                best_params.get('weight_effect', best_params_default['weight_effect']),
                best_params.get('weight_urgency', best_params_default['weight_urgency']),
                best_params.get('weight_uncertainty', best_params_default['weight_uncertainty']),
                best_params.get('confidence_level', best_params_default['confidence_level'])
            )
            end_time_per_rule = datetime.now()
            time_taken_per_rule = end_time_per_rule - start_time_per_rule
            time_taken_per_rule_dict.setdefault(resources, {})[rule_name] = time_taken_per_rule.total_seconds()

            if hyper_opt_argv:
                results_dict_hyperopt.setdefault(log_name, {}).update(rule_treated_cases_dict)
            else:
                results_dict_no_hyperopt.setdefault(log_name, {}).update(rule_treated_cases_dict)

    if hyper_opt_argv:
        results_dict_all["Optimized"] = results_dict_hyperopt
        results_type = "Optimized"
    else:
        results_dict_all["NoOptimization"] = results_dict_no_hyperopt
        results_type = "NoOptimization"

    save_results(results_dict_all, best_params_dict, time_taken_dict, best_trial_dict, resources, treatment_duration, results_type)

    # Save time taken per rule to a separate file
    with open(f"{results_directory}/{results_type}_{resources}_{treatment_duration}_time_taken_per_rule.pkl", "wb") as f:
        pickle.dump(time_taken_per_rule_dict, f)

def save_results(results_dict_all, best_params_dict, time_taken_dict, best_trial_dict, resources, treatment_duration, results_type):
    #results_directory = f"./results_directory"  # Replace with the actual directory
    import os
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    import pickle
    with open(f"{results_directory}/{results_type}_{resources}_{treatment_duration}_results_dict_all.pkl", "wb") as f:
        pickle.dump(results_dict_all, f)

    with open(f"{results_directory}/{results_type}_{resources}_{treatment_duration}_best_params.pkl", "wb") as f:
        pickle.dump(best_params_dict, f)

    with open(f"{results_directory}/{results_type}_{resources}_{treatment_duration}_time_taken_dict.pkl", "wb") as f:
        pickle.dump(time_taken_dict, f)

    with open(f"{results_directory}/{results_type}_{resources}_{treatment_duration}_best_trial_dict.pkl", "wb") as f:
        pickle.dump(best_trial_dict, f)


# Function to get the max value from a column in df
def get_max_ub(df, col):
    return df[col].max()

# Function to get the max value from a column in df
def get_min_ub(df, col):
    return df[col].min()

# Function to get the max value from a column in df
def get_mean_ub(df, col):
    return df[col].mean()


if __name__ == "__main__":
    resources_list = [resourcesAll]
    filter_obj = FilterCases()
    

    # Define other necessary variables and functions
    best_params_default = {
        'neg_proba': 0.5, 'proba_conformal': 0.0, 'proba_uncer': 0.5, 'te': 0.0,
        'causal_uncer': 0.0, 'iw': 10, 'confidence_level': 0.9,
        'weight_need': 1, 'weight_effect': 1, 'weight_urgency': 1, 'weight_uncertainty': 1,
    }

    results_dict_all = {"Optimized": {}, "NoOptimization": {}}
    time_taken_dict = {}
    best_trial_dict = {}
    best_params_dict = {}
    rule_treated_cases_dict = {}

    confidence_level = best_params_default['confidence_level']
    alpha_value = np.round((1 - confidence_level), 1)

    max_ub_te = get_max_ub(valid_all, f"upper_counterfactual_1_alpha_{alpha_value}")
    max_ub_iw = get_max_ub(valid_all, f"upper_bound_{alpha_value}")
    min_ub_iw = get_min_ub(valid_all, f"upper_bound_{alpha_value}")
    mean_ub_iw = get_mean_ub(valid_all, f"upper_bound_{alpha_value}")

    best_params_default['iw']= mean_ub_iw


    hyper_opt_list = [True]
    for hyper_opt_argv in hyper_opt_list:
        run_testing_phase(resources_list, filter_obj, hyper_opt_argv, best_params_default)













