import pandas as pd
import numpy as np
import argparse
from sys import argv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from causal_estimators.wtt_estimator import wtt_estimator
#from Predictive_models.utils import *

# case_id_col = "Case ID"
# activity_col = "Activity"
# resource_col = "org:resource"
# timestamp_col = "time:timestamp"

# treatment = 'treatment'
# outcome = 'outcome'

# dynamic_cat_cols = ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition']
# static_cat_cols = ['ApplicationType', 'LoanGoal']
# dynamic_num_cols = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', "open_cases", "month", "weekday", "hour",
#                     "timesincelastevent", "timesincecasestart", "timesincemidnight",'t_started']
# static_num_cols = ['RequestedAmount', 'CreditScore', 'timesincefirstcase', 'treatment', 'outcome','time_of_treatment']

# cat_cols = dynamic_cat_cols + static_cat_cols
# num_cols = dynamic_num_cols + static_num_cols


# def get_args():
#     parser = argparse.ArgumentParser(description="when_to_treat")

#     # dataset
#     # parser.add_argument("--debug", type=str_to_bool, default=False)
#     parser.add_argument("--data", type=str, default="bpic17",choices=["bpic17","bpic19"])
#     parser.add_argument("--estimator", type=str, default='CausalForest')
#     parser.add_argument("--propensity_model", type=str, default='LogisticRegression')
#     parser.add_argument("--outcome_model", type=str, default='RandomForestClassifier')
#     parser.add_argument("--conf_thresh", type=float, default=0.1)
#     # parser.add_argument("--is_forest", type=bool, default=False)

#     return parser
import sys
import os
import time
sys.path.insert(1, "/home/centos/phd/3rdyear/2nd/myCode/common_files")

from DatasetManager import DatasetManager
import EncoderFactory

# print("Read input...")
#dataset_name = argv[1]  # prepared_bpic2017
#print(dataset_name)
# optimal_params_filename = argv[2]  # params_dir
#results_dir = argv[2]  # 
#print(results_dir)

#prefixes_dir = argv[3]


# # create results directory
# if not os.path.exists(os.path.join(results_dir)):
#     os.makedirs(os.path.join(results_dir))


def get_args():
    parser = argparse.ArgumentParser(description="when_to_treat")

    # dataset
    #parser.add_argument("--debug", type=str_to_bool, default=False)
    parser.add_argument("--dataset_name", type=str, default="trafficFines",choices=["trafficFines", "bpic2012","bpic2017"])
    parser.add_argument("--estimator", type=str, default='CausalForest')
    parser.add_argument("--propensity_model", type=str, default='LogisticRegression')
    parser.add_argument("--outcome_model", type=str, default='RandomForestClassifier')
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--results_dir", type=str, default=".results/causal/")
    # parser.add_argument("--is_forest", type=bool, default=False)

    return parser

def main(args):
    print(args.dataset_name)
    # print("Read input...")
    #dataset_name = argv[1]  # prepared_bpic2017
    #print(dataset_name)
    # optimal_params_filename = argv[2]  # params_dir
    #results_dir = argv[2]  # 
    #print(results_dir)



    if not os.path.exists(os.path.join(args.results_dir)):
        os.makedirs(os.path.join(args.results_dir))

        
    print("Reading data...")
    start = time.time()

    print("./results/causal/%s/train_df_CATE_%s.csv"%(args.dataset_name, args.dataset_name))
    df_train = pd.read_csv("./results/causal/%s/train_df_CATE_%s.csv"%(args.dataset_name, args.dataset_name), sep=';')
    df_test = pd.read_csv("./results/causal/%s/test_df_CATE_%s.csv"%(args.dataset_name, args.dataset_name), sep=';')
    df_valid = pd.read_csv("./results/causal/%s/valid_df_CATE_%s.csv"%(args.dataset_name, args.dataset_name), sep=';')

    print('preparing data')
    features = df_train.drop(["Outcome", "Treatment",], axis=1)
    features_test = df_test.drop(["Outcome", "Treatment", ], axis=1)
    features_valid = df_valid.drop(["Outcome", "Treatment", ], axis=1)

    Y = df_train["Outcome"].to_numpy()
    T = df_train["Treatment"].to_numpy()


    ###### Standardisation ######
    scaler = MinMaxScaler()
    W = scaler.fit_transform(features[[c for c in features.columns]].to_numpy())
    X = scaler.fit_transform(features[[c for c in features.columns]].to_numpy())

    X_test = scaler.fit_transform(features_test[[c for c in features_test.columns]].to_numpy())
    X_valid = scaler.fit_transform(features_valid[[c for c in features_valid.columns]].to_numpy())

    df_results_test = df_test
    df_results_valid = df_valid

    wtt = wtt_estimator(args)
    wtt.initialize_forest()
    wtt.fit_forest(X,T,Y)

    lower_test,upper_test = wtt.get_te_withCI(X_test)
    lower_valid,upper_valid = wtt.get_te_withCI(X_valid)

    df_results_test['upper_cate'] = upper_test
    df_results_test['lower_cate'] = lower_test
    wtt.save_results(df_results_test)


    df_results_valid['upper_cate'] = upper_valid
    df_results_valid['lower_cate'] = lower_valid
    wtt.save_results(df_results_valid)

    df_results_test.to_csv(os.path.join("./results/causal/%s/"%args.dataset_name, 'df_results_test_lower_upper_cate_%s.csv'%args.dataset_name),  index=False, sep=';')
    df_results_valid.to_csv(os.path.join("./results/causal/%s/"%args.dataset_name, 'df_results_valid_lower_upper_cate_%s.csv'%args.dataset_name),  index=False, sep=';')


    print('@report: Ended')


if __name__ == '__main__':
    


    print("@report: Started")
    main(get_args().parse_args())
