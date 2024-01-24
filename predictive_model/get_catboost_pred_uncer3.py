# # some_file.py
# import sys
# sys.path.insert(1, "/home/centos/phd/4thyear/RL-prescriptive-monitoring/common_files")

# from DatasetManager import DatasetManager
# import EncoderFactory
# import os
# import time
# from sys import argv
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.pipeline import FeatureUnion
# from catboost import CatBoostClassifier
# from concurrent.futures import ThreadPoolExecutor
# from DatasetManager import DatasetManager
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder


import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor
from sys import argv

# print("Read input...")
dataset_name = argv[1]  # prepared_bpic2017
# optimal_params_filename = argv[2]  # params_dir
results_dir = argv[2]  # results_dir

en_size = int(argv[3])  # size of the ensemble
print(f"Ensemble size is: {en_size}")



# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))




# Function to check if a file exists
def file_exists(file_path):
    return os.path.exists(file_path)



file_path = "/home/centos/phd/4thyear/RL-prescriptive-monitoring/prepared_data/%s/"%dataset_name

train_data = pd.read_csv(os.path.join(file_path, 'train_encoded_%s.csv'%dataset_name), low_memory=False,  sep=';')
print(train_data.columns)
valid_data = pd.read_csv(os.path.join(file_path, 'valid_encoded_%s.csv'%dataset_name), low_memory=False,  sep=';')
print(valid_data.columns)
test_data = pd.read_csv(os.path.join(file_path, 'test_encoded_%s.csv'%dataset_name), low_memory=False,  sep=';')
print(test_data.columns)


cat_feat_idx = np.where(train_data.dtypes == object)[0]
print(cat_feat_idx)

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
dt_preds_valid.to_csv(os.path.join(results_dir, "preds_valid_%s.csv" % dataset_name), sep=";", index=False)
dt_preds_test.to_csv(os.path.join(results_dir, "preds_test_%s.csv" % dataset_name), sep=";", index=False)
