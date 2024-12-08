from plotly.graph_objs.volume.caps import X

# some_file.py
import sys
#import tensorflow as tf
import datetime

#from causallift import Ca
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "/home/centos/phd/3rdyear/2nd/myCode/common_files")

from DatasetManager import DatasetManager
import EncoderFactory

# DatasetManager import DatasetManager

import causallift
from causallift import CausalLift

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
from sys import argv
import pickle

import subprocess
import sys


import causallift
print(causallift.__version__)
from causallift import CausalLift

# print("Read input...")
dataset_name = argv[1]  # prepared_bpic2017
# optimal_params_filename = argv[2]  # params_dir
results_dir = argv[2]  # 

prefixes_dir = argv[3]

# en_size = int(argv[4])  # size of the ensemble
# print(f"Ensemble size is: {en_size}")

calibrate = False
split_type = "temporal"
oversample = False
calibration_method = "beta"

train_ratio = 0.5
val_ratio = 0.5

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

print("Reading data...")
start = time.time()

# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()


min_prefix_length = 1
max_prefix_length = int(
    np.ceil(data.groupby(dataset_manager.case_id_col).size().quantile(0.9))
)

cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': True,
                    'dataset_name':dataset_name,
                    "results_dir":results_dir,
                    "model":"causalLift"}


dt_train_prefixes = pd.read_csv(os.path.join(prefixes_dir, 'dt_train_prefixes_%s.csv'%dataset_name),  low_memory=False, sep=';')
dt_test_prefixes = pd.read_csv(os.path.join(prefixes_dir, 'dt_test_prefixes_%s.csv'%dataset_name),  low_memory=False,  sep=';')
dt_val_prefixes = pd.read_csv(os.path.join(prefixes_dir, 'dt_val_prefixes_%s.csv'%dataset_name),  low_memory=False, sep=';')





# encode all prefixes
feature_combiner = FeatureUnion(
    [
        (method, EncoderFactory.get_encoder(method, **cls_encoder_args))
        for method in ["static", "agg"]
    ]
)
print("Start encoding...")

def encode_data(prefixes, type="train"):    
    x = feature_combiner.fit_transform(prefixes)
    y = dataset_manager.get_label_numeric(prefixes)
    t = dataset_manager.get_treatment_numeric(prefixes)
    data = pd.concat(
        [
            pd.DataFrame(x),
            pd.DataFrame(y),
            pd.DataFrame(t)
        ],
        axis=1,
    )  
    data.to_pickle(os.path.join(results_dir, "%s_treatment.pkl" % type))

    return data


train_data = encode_data(dt_train_prefixes, type="train")
test_data = encode_data(dt_test_prefixes, type="test")
valid_data = encode_data(dt_val_prefixes, type="valid")




print("Read encoded data...")
df_agg = pd.read_csv(os.path.join(results_dir, 'dt_transformed_agg_%s.csv'%dataset_name),  sep=';')
df_static = pd.read_csv(os.path.join(results_dir, 'dt_transformed_static_%s.csv'%dataset_name),  sep=';')


static_agg_df = pd.concat([df_static, df_agg], axis=1)
cat_feat_idx = np.where(static_agg_df.dtypes != float)[0]

print(len(static_agg_df.columns))


#  rename columns
train_data.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]
test_data.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]
valid_data.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]

cols = train_data.columns
train_data[cols] = train_data[cols].apply(pd.to_numeric, errors='coerce')
test_data[cols] = test_data[cols].apply(pd.to_numeric, errors='coerce')
valid_data[cols] = valid_data[cols].apply(pd.to_numeric, errors='coerce')

print(f"\n crosstab train:\n{pd.crosstab(train_data['Outcome'], train_data['Treatment'], margins = True)}\n")
print(f"\n crosstab test:\n{pd.crosstab(test_data['Outcome'], test_data['Treatment'], margins = True)}\n")
print(f"\n crosstab valid:\n{pd.crosstab(valid_data['Outcome'], valid_data['Treatment'], margins = True)}\n")

import xgboost as xgb

train_data_m = xgb.DMatrix(train_data)
test_data_m = xgb.DMatrix(test_data)
valid_data_m = xgb.DMatrix(valid_data)

print('\n[Estimate propensity scores for Inverse Probability Weighting.]')
cl_train_test = CausalLift(train_data, test_data, enable_ipw=True, verbose=2,  

 uplift_model_params={'cv':
3, 'estimator': 'xgboost.XGBClassifier', 'n_jobs': -1,
'param_grid': {'base_score': [0.5], 'booster': ['gbtree'],
'colsample_bylevel': [1], 'colsample_bytree': [1], 'gamma': [0],
'learning_rate': [0.1], 'max_delta_step': [0], 'max_depth': [3],
'min_child_weight': [1], 'missing': [1], 'n_estimators':
[100], 'n_jobs': [-1], 'nthread': [None], 'objective':
['binary:logistic'], 'random_state': [0], 'reg_alpha': [0],
'reg_lambda': [1], 'scale_pos_weight': [1], 'subsample': [1],
'verbose': [0]}, 'return_train_score': False, 'scoring': None,
'search_cv': 'sklearn.model_selection.GridSearchCV'},

propensity_model_params={'cv': 3, 'estimator':
'sklearn.linear_model.LogisticRegression', 'n_jobs': -1,
'param_grid': {'C': [0.1, 1, 10], 'class_weight': [None], 'dual':
[False], 'fit_intercept': [True], 'intercept_scaling': [1],
'max_iter': [1000], 'multi_class': ['ovr'], 'n_jobs': [1], 'penalty':
['l1', 'l2'], 'random_state': [0], 'solver': ['liblinear'], 'tol':
[0.0001], 'warm_start': [False]}, 'return_train_score': False,
'scoring': None, 'search_cv':
'sklearn.model_selection.GridSearchCV'})
cl_train_val = CausalLift(train_data, valid_data, enable_ipw=True, verbose=2,  

 uplift_model_params={'cv':
3, 'estimator': 'xgboost.XGBClassifier', 'n_jobs': -1,
'param_grid': {'base_score': [0.5], 'booster': ['gbtree'],
'colsample_bylevel': [1], 'colsample_bytree': [1], 'gamma': [0],
'learning_rate': [0.1], 'max_delta_step': [0], 'max_depth': [3],
'min_child_weight': [1], 'missing': [1], 'n_estimators':
[100], 'n_jobs': [-1], 'nthread': [None], 'objective':
['binary:logistic'], 'random_state': [0], 'reg_alpha': [0],
'reg_lambda': [1], 'scale_pos_weight': [1], 'subsample': [1],
'verbose': [0]}, 'return_train_score': False, 'scoring': None,
'search_cv': 'sklearn.model_selection.GridSearchCV'},

propensity_model_params={'cv': 3, 'estimator':
'sklearn.linear_model.LogisticRegression', 'n_jobs': -1,
'param_grid': {'C': [0.1, 1, 10], 'class_weight': [None], 'dual':
[False], 'fit_intercept': [True], 'intercept_scaling': [1],
'max_iter': [1000], 'multi_class': ['ovr'], 'n_jobs': [1], 'penalty':
['l1', 'l2'], 'random_state': [0], 'solver': ['liblinear'], 'tol':
[0.0001], 'warm_start': [False]}, 'return_train_score': False,
'scoring': None, 'search_cv':
'sklearn.model_selection.GridSearchCV'})


print('\n[Create 2 models for treatment and untreatment and estimate CATE (Conditional Average Treatment Effects)]')
train_df, test_df = cl_train_test.estimate_cate_by_2_models()

print('\n[Show CATE for train dataset]')
#display(train_df)
train_df.to_csv(os.path.join(results_dir, 'train_df_CATE_%s.csv'%dataset_name),  index=False, sep=';')

print('\n[Show CATE for test dataset]')
#display(test_df)
test_df.to_csv(os.path.join(results_dir, 'test_df_CATE_%s.csv'%dataset_name),  index=False, sep=';')



train_df, valid_df = cl_train_val.estimate_cate_by_2_models()
print('\n[Show CATE for valid dataset]')
#display(valid_df)
valid_df.to_csv(os.path.join(results_dir, 'valid_df_CATE_%s.csv'%dataset_name),  index=False, sep=';')



print('\n[Estimate the effect of recommendation based on the uplift model]')
estimated_effect_df = cl_train_test.estimate_recommendation_impact()


print('\n[Estimate the effect of recommendation based on the uplift model]')
estimated_effect_df = cl_train_val.estimate_recommendation_impact()
