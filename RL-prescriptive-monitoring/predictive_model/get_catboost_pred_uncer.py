from plotly.graph_objs.volume.caps import X

# some_file.py
import sys
import tensorflow as tf
import datetime

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "/home/centos/phd/3rdyear/2nd/myCode/common_files")


from DatasetManager import DatasetManager
import EncoderFactory

# DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
from sys import argv
import pickle

from catboost import Pool, CatBoostRegressor, CatBoostClassifier

# print("Read input...")
dataset_name = argv[1]  # prepared_bpic2017
# optimal_params_filename = argv[2]  # params_dir
results_dir = argv[2]  # results_dir

en_size = int(argv[3])  # size of the ensemble
print(f"Ensemble size is: {en_size}")

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
    np.ceil(data.groupby(dataset_manager.case_id_col).size().quantile(1))
)

# max_prefix_length = int(
#     np.ceil(data.groupby(dataset_manager.case_id_col).size().quantile(0.9))
# )

cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': True,
                    'dataset_name':dataset_name,
                    "results_dir":results_dir}


# split into training and test
if split_type == "temporal":
    train, test = dataset_manager.split_data_strict(data, train_ratio, split=split_type)
else:
    train, test = dataset_manager.split_data(data, train_ratio, split=split_type)

test, val = dataset_manager.split_val(test, val_ratio)
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")
print(f"val shape: {val.shape}")
print(train.columns)
print(len(set(train['Case ID'])))
print(len(set(test['Case ID'])))
print(len(set(val['Case ID'])))

#raise SystemExit()


# generate data where each prefix is a separate instance
dt_train_prefixes = dataset_manager.generate_prefix_data(
    train, min_prefix_length, max_prefix_length
)
dt_train_prefixes.to_pickle(os.path.join(results_dir, 'dt_train_prefixes_%s.pkl'%dataset_name))
dt_train_prefixes.to_csv(os.path.join(results_dir, 'dt_train_prefixes_%s.csv'%dataset_name),  index=False, sep=';')

dt_test_prefixes = dataset_manager.generate_prefix_data(
    test, min_prefix_length, max_prefix_length
)
dt_test_prefixes.to_pickle(os.path.join(results_dir, 'dt_test_prefixes_%s.pkl'%dataset_name))
dt_test_prefixes.to_csv(os.path.join(results_dir, 'dt_test_prefixes_%s.csv'%dataset_name),  index=False, sep=';')

dt_val_prefixes = dataset_manager.generate_prefix_data(
    val, min_prefix_length, max_prefix_length
)
dt_val_prefixes.to_pickle(os.path.join(results_dir, 'dt_val_prefixes_%s.pkl'%dataset_name))
dt_val_prefixes.to_csv(os.path.join(results_dir, 'dt_val_prefixes_%s.csv'%dataset_name),  index=False, sep=';')


print(f"dt_train_prefixes: {dt_train_prefixes.shape}")
print(f"dt_test_prefixes shape: {dt_test_prefixes.shape}")
print(f"dt_val_prefixes shape: {dt_val_prefixes.shape}")
print(train.columns)
print(len(set(dt_train_prefixes['Case ID'])))
print(len(set(dt_test_prefixes['Case ID'])))
print(len(set(dt_val_prefixes['Case ID'])))

#raise SystemExit()


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
    data = pd.concat(
        [
            pd.DataFrame(x),
            pd.DataFrame(y),
        ],
        axis=1,
    )  
    data.to_pickle(os.path.join(results_dir, "%s_.pkl" % type))

    return data


train_data = encode_data(dt_train_prefixes, type="train")
test_data = encode_data(dt_test_prefixes, type="test")
valid_data = encode_data(dt_val_prefixes, type="valid")




print("Read encoded data...")
df_agg = pd.read_csv(os.path.join(results_dir, 'dt_transformed_agg_%s.csv'%dataset_name), low_memory=False,  sep=';')
df_static = pd.read_csv(os.path.join(results_dir, 'dt_transformed_static_%s.csv'%dataset_name), low_memory=False,  sep=';')


static_agg_df = pd.concat([df_static, df_agg], axis=1)
cat_feat_idx = np.where(static_agg_df.dtypes != float)[0]

#  rename columns
train_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]
test_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]
valid_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]

y_train = train_data['Outcome']
X_train = train_data.drop(['Outcome', ], axis=1)

y_valid = valid_data['Outcome']
X_valid = valid_data.drop(['Outcome'], axis=1)

y_test = test_data['Outcome']
X_test = test_data.drop(['Outcome'], axis=1)

print("Create modle...")
print(f"Cat_feat_idx: {cat_feat_idx}")


# Ensemble of CatBoost
class Ensemble(object):

    def __init__(self, esize=10, iterations=1000, lr=0.1, random_strength=0, border_count=128, depth=6, seed=100, best_param=None):

        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr  # from tunning
        self.random_strength = random_strength
        self.border_count = border_count
        self.best_param = best_param
        self.ensemble = []
        for e in range(self.esize):
            model = CatBoostClassifier(iterations=self.iterations,
                                       depth=self.depth,
                                       border_count=self.border_count,
                                       random_strength=self.random_strength,
                                       loss_function='Logloss',  # -ve likelihood
                                       verbose=False,
                                       bootstrap_type='Bernoulli',
                                       posterior_sampling=True,
                                       eval_metric='AUC',
                                       use_best_model=True,
                                       langevin=True,
                                       random_seed=self.seed + e)
            self.ensemble.append(model)

    def fit(self, X_train, y_train, cat_feat_idx, eval_set=None):
        count = 1
        for m in self.ensemble:            
            callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=results_dir, histogram_freq=1)
            print(f"\nFitting model...{count}")
            count+=1
            print(set(y_train))
            print(y_train.dtype)
            print(set(y_valid))
            print(y_valid.dtype)
            m.fit(X_train, y=y_train, cat_features=cat_feat_idx,  eval_set=(X_valid, y_valid))
            print("best iter ", m.get_best_iteration())
            print("best score ", m.get_best_score())

    def predict_proba(self, x):
        probs = []
        for m in self.ensemble:
            prob = m.predict_proba(x)
            probs.append(prob)
        probs = np.stack(probs)
        return probs

    def predict(self, x):
        preds = []
        for m in self.ensemble:
            pred = m.predict(x)
            preds.append(pred)
        preds = np.stack(preds)
        return preds
    
    def get_reliability(self, preds, probs):
        print("\nget_reliability\n")
        #print(self)
        preds, probs = preds, probs #self.get_preds(X)
        # print(preds)
        # print(probs)
        # preds = np.transpose(preds)
        deviation = (1 - preds)/1
        print(deviation)
        reliability = np.count_nonzero(np.transpose(deviation), axis=1)/deviation.shape[0]

        return reliability, deviation


# eoe: Total uncer: entropy of the avg predictions
def entropy_of_expected(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


# Data uncer: avg(entropy of indviduals)
def expected_entropy(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)


# Knowledge uncer
def mutual_information(probs, epsilon):
    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    return eoe - exe  # knowldge_ucer = total_uncer - data_uncer


def ensemble_uncertainties(probs, epsilon=1e-10):
    #print(f"Probs: {np.max(probs)}")
    print(f"Ensemble size: {len(probs)}\n")
    mean_probs = np.mean(probs, axis=0)  # avg ensamble prediction
    conf = np.max(mean_probs, axis=1)  # max avg ensamble prediction: predicted class

    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    mutual_info = eoe - exe

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,  # Total uncer: entropy of the avg predictions
                   'expected_entropy': exe,  # Data uncer: avg(entropy of indviduals)
                   'mutual_information': mutual_info,  # Knowledge uncer
                   }
    print(f"total_uncer: {eoe}")
    print(f"len total_uncer: {len(eoe)}\n")

    print(f"Data_uncer: {exe}")
    print(f"len Data_uncer: {len(exe)}\n")

    print(f"Knowldge_uncer: {mutual_info}")
    print(f"len Knowldge_uncer: {len(mutual_info)}\n")

    return uncertainty




ens = Ensemble(esize=en_size, iterations=1000, lr=0.1, depth=6, seed=2, random_strength = 100,)
ens.fit(X_train, y_train, cat_feat_idx, eval_set=(X_valid, y_valid))

probs_train = ens.predict_proba(X_train)
probs_test = ens.predict_proba(X_test)
probs_valid = ens.predict_proba(X_valid)

preds_train_e = ens.predict(X_train)
preds_test_e = ens.predict(X_test)
preds_valid_e = ens.predict(X_valid)

#reliability_train = 


probs_train_mean = np.mean(ens.predict_proba(X_train), axis=0)
probs_test_mean = np.mean(ens.predict_proba(X_test), axis=0)
probs_valid_mean = np.mean(ens.predict_proba(X_valid), axis=0)

uncerts_train = ensemble_uncertainties(probs_train)
uncerts_test = ensemble_uncertainties(probs_test)
uncerts_valid = ensemble_uncertainties(probs_valid)


print("Predict train...")
preds_train_prob_1 = probs_train_mean[:, 1]
preds_train_prob_0 = probs_train_mean[:, 0]
preds_train = np.array(pd.DataFrame(preds_train_e).mode().iloc[0].astype(int))

#np.array(pd.DataFrame(preds).mode().iloc[0].astype(int))

print("Predict test...")
preds_test_prob_1 = probs_test_mean[:, 1]
preds_test_prob_0 = probs_test_mean[:, 0]
preds_test = np.array(pd.DataFrame(preds_test_e).mode().iloc[0].astype(int))

print("Predict valid")
preds_valid_prob_1 = probs_valid_mean[:, 1]
preds_valid_prob_0 = probs_valid_mean[:, 0]
preds_valid = np.array(pd.DataFrame(preds_valid_e).mode().iloc[0].astype(int))

print("Save results")
# write train set predictions
dt_preds_train = pd.DataFrame({"predicted_proba_0": preds_train_prob_0,
                               "predicted_proba_1": preds_train_prob_1,
                               "predicted": preds_train,
                               "actual": y_train,
                               "total_uncer": uncerts_train['entropy_of_expected'],
                              "data_uncer": uncerts_train['expected_entropy'],
                                "knowledge_uncer": uncerts_train["mutual_information"],
                                "confidence": uncerts_train["confidence"] })
dt_preds_train.to_pickle(os.path.join(results_dir, "preds_train_%s.pkl" % dataset_name))

# write test set predictions
dt_preds_test = pd.DataFrame({"predicted_proba_0": preds_test_prob_0,
                              "predicted_proba_1": preds_test_prob_1,
                              "predicted": preds_test,
                              "actual": y_test,
                               "total_uncer": uncerts_test['entropy_of_expected'],
                              "data_uncer": uncerts_test['expected_entropy'],
                                "knowledge_uncer": uncerts_test["mutual_information"],
                                "confidence": uncerts_test["confidence"] })
dt_preds_test.to_pickle(os.path.join(results_dir, "preds_test_%s.pkl" % dataset_name))

# write valid set predictions
dt_preds_valid = pd.DataFrame({"predicted_proba_0": preds_valid_prob_0,
                               "predicted_proba_1": preds_valid_prob_1,
                               "predicted": preds_valid,
                               "actual": y_valid,
                               "total_uncer": uncerts_valid['entropy_of_expected'],
                              "data_uncer": uncerts_valid['expected_entropy'],
                                "knowledge_uncer": uncerts_valid["mutual_information"],
                                "confidence": uncerts_valid["confidence"]})
dt_preds_valid.to_pickle(os.path.join(results_dir, "preds_valid_%s.pkl" % dataset_name))

print("write train-val set predictions CSV")
dt_preds_train.to_csv(os.path.join(results_dir, "preds_train_%s.csv" % dataset_name), sep=";", index=False)
dt_preds_valid.to_csv(os.path.join(results_dir, "preds_val_%s.csv" % dataset_name), sep=";", index=False)
dt_preds_test.to_csv(os.path.join(results_dir, "preds_test_%s.csv" % dataset_name), sep=";", index=False)
