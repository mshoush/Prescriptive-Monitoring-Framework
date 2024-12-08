#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys

pd.set_option("display.max_columns", None)

sys.path.insert(1, "/home/centos/phd/3rdyear/2nd/myCode/common_files")

from DatasetManager import DatasetManager


from sklearn.preprocessing import *
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import RobustScaler
import os
from sys import argv


dataset_name = argv[1]

dataset_manager = DatasetManager(dataset_name)


# In[2]:


# read conformal survival data
conformal_survival = pd.read_csv("./results/conformal_survival/%s/test_data_survival_conformal_%s.csv"%(dataset_name, dataset_name),
                                 sep=';' )

# read conformal survival data
conformal_ite = pd.read_csv(
    "./results/conformal_causal/%s/test2_conformalizedTE_%s.csv"%(dataset_name, dataset_name),
    sep=",",
    # thousands=",",
)

#print(conformal_survival.columns)
#print(conformal_ite.columns)
#print(f"conformal_survival: {conformal_survival.shape}, {len(set(conformal_survival['case_id']))}")
#print(f"conformal_ite: {conformal_ite.shape}")
#raise SystemExit()



cols_to_use = [
    "lower_time_to_event_adaptive",
    "lower_time_to_event_adaptive_QR",
    "lower_time_to_event_naive",
    "predicted_time_to_event",
    "time_to_event_m",
    "upper_time_to_event_adaptive",
    "upper_time_to_event_adaptive_QR",
    "upper_time_to_event_naive",
    "timestamp",
    "activity",
    "case_id",
]
cols_to_use


# In[8]:


dfNew = pd.merge(
    conformal_ite,
    conformal_survival[cols_to_use],
    left_index=True,
    right_index=True,
    how="outer",
)

#print(f"dfNew: {dfNew.shape}, {len(set(dfNew['case_id']))}")
#print(f"conformal_survival: {conformal_survival.shape}, {len(set(conformal_survival['case_id']))}")
#print(f"conformal_ite: {conformal_ite.shape}")
#raise SystemExit()


try:
    dfNew.rename(columns={"time_to_event_m_y": "time_to_event_m"}, inplace=True)
except: 
    pass
print(dfNew.columns)


import os
pd.set_option('display.max_columns', None)

names = ['adaptive', "classBalanced", "naive"]
datasets = [dataset_name]
case_id_col = "case_id"
timestamp_col = "timestamp"


# In[11]:


for name in names:
    for dataset in datasets:
        results_dir = "./results/compiled_results_for_RL/%s/" % dataset
        # create results directory
        if not os.path.exists(os.path.join(results_dir)):
            os.makedirs(os.path.join(results_dir))

        print(f"\ndataset: {dataset}\nname: {name}\n")

        csv_file = "./results/conformal/%s/df_final_%s_all_%s.csv" % (
            dataset,
            name,
            dataset,
        )
        df = pd.read_csv(csv_file, sep=";")
        df_obj = df.select_dtypes(["object"])
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        print(df.shape)

        te_df = df.copy()
       # print(f"te_df: {te_df.shape}, {len(set(te_df['case_id']))}")
       # print(f"dfNew: {dfNew.shape}, {len(set(dfNew['case_id']))}")
       # print(f"conformal_survival: {conformal_survival.shape}, {len(set(conformal_survival['case_id']))}")
       # print(f"conformal_ite: {conformal_ite.shape}")
       # raise SystemExit()

        #for col in te_df.select_dtypes(include="O").columns:
        #    print(col)
        #    te = TargetEncoder()
        #    te_df[col] = te.fit_transform(te_df[col], te_df.actual)
        te_df["y1"] = np.where(te_df["Proba_if_Treated"] > 0.5, 1, 0)
        te_df["y0"] = np.where(te_df["Proba_if_Untreated"] > 0.5, 1, 0)

        probs = np.array([te_df["predicted_proba_0"], te_df["predicted_proba_1"]])
        preds = np.array([te_df["predicted"]])
        deviation = (1 - preds) / 1
        reliability = (
            np.count_nonzero(np.transpose(deviation), axis=1) / deviation.shape[0]
        )

        te_df["deviation"] = deviation.ravel()
        te_df["reliability"] = reliability

        te_df["ordinal_case_ids"] = pd.factorize(te_df["case_id"])[0] + 1
        #print(f"te_df: {te_df.shape}, {len(set(te_df['case_id']))}")
        #print(f"dfNew: {dfNew.shape}, {len(set(dfNew['case_id']))}")
        #print(f"conformal_survival: {conformal_survival.shape}, {len(set(conformal_survival['case_id']))}")
        #print(f"conformal_ite: {conformal_ite.shape}")
        #raise SystemExit()


        cols_to_use = [
            "upper_counterfactual",
            "upper_exact",
            "upper_inexact",
            "upper_naive",
            "upper_time_to_event_adaptive",
            "upper_time_to_event_adaptive_QR",
            "upper_time_to_event_naive",
            "time_to_event_m",
            "lower_counterfactual",
            "lower_exact",
            "lower_inexact",
            "lower_naive",
            "lower_time_to_event_adaptive",
            "lower_time_to_event_adaptive_QR",
            "lower_time_to_event_naive",
            "timestamp",
            "activity",
        ]

        dfNew_te_df = pd.merge(
            te_df, dfNew[cols_to_use], left_index=True, right_index=True, how="outer",
        )
        dfNew_te_df.rename(columns={"timestamp_y": "orig_timestamp"}, inplace=True)
        dfNew_te_df.rename(columns={"timestamp_x": "encoded_timestamp"}, inplace=True)

        dfNew_te_df.rename(columns={"activity_y": "orig_activity"}, inplace=True)
        dfNew_te_df.rename(columns={"activity_x": "encoded_activity"}, inplace=True)

        def extract_case_length(group):
            group["case_length"] = group["prefix_nr"].max()
            return group

        df = (
            dfNew_te_df.groupby(case_id_col, as_index=False)
            .apply(extract_case_length)
            .reset_index(drop=True)
        )

        #             df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1))
        pd.set_option('display.max_rows', None)
        def sort_prefixes(group):
            #group =  group.reset_index(drop=True)
            #group = group.drop_duplicates()
            #print(group["prefix_nr"])
            max_prefix = group.shape[0] #group["prefix_nr"].max()
            #print(group.shape)
            #print(max_prefix)
            group["prefix_nr"] = list(range(1, max_prefix + 1))
            #print(max_prefix)
            return group

        df = (
            df.groupby(case_id_col, as_index=False)
            .apply(sort_prefixes)
            .reset_index(drop=True)
        )

        # df["ordinal_case_ids"] = pd.factorize(te_df["case_id"])[0] + 1i
        #print(df.columns)
        #print(f"df: {df.shape}, {len(set(df['case_id']))}")
        #print(f"te_df: {te_df.shape}, {len(set(te_df['case_id']))}")
        #print(f"dfNew: {dfNew.shape}, {len(set(dfNew['case_id']))}")
        #print(f"conformal_survival: {conformal_survival.shape}, {len(set(conformal_survival['case_id']))}")
        #print(f"conformal_ite: {conformal_ite.shape}")
        #raise SystemExit()

        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        # assuming df is your DataFrame
        scaler = MinMaxScaler()
        cols = ['lower_time_to_event_adaptive', 'lower_time_to_event_adaptive_QR',
                       'lower_time_to_event_naive', 'upper_time_to_event_adaptive', 'upper_time_to_event_adaptive_QR',
                              'upper_time_to_event_naive', 'time_to_event_m', ]
        for col in cols:
            if df[col].dtype == 'float64':
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

        alpha_columns = [col for col in df.columns if col.startswith("alpha")]
        alpha_columns
        for col in alpha_columns:
            df[str(col) + "_encoded"] = df[col].apply(lambda x: 1 if x == "[1]" else 0)


        print(df.columns)
        print(f"df: {df.shape}, {len(set(df['case_id']))}")
        print(f"te_df: {te_df.shape}, {len(set(te_df['case_id']))}")
        print(f"dfNew: {dfNew.shape}, {len(set(dfNew['case_id']))}")
        print(f"conformal_survival: {conformal_survival.shape}, {len(set(conformal_survival['case_id']))}")
        print(f"conformal_ite: {conformal_ite.shape}")
        #raise SystemExit()


        df.to_csv(
            os.path.join(results_dir, "ready_to_use_%s_%s.csv" % (name, dataset)),
            index=False,
            sep=";",
        )


# In[12]:


#df = pd.read_csv(
#    "./results/compiled_results_for_RL/bpic2012/ready_to_use_adaptive_bpic2012.csv",
#    sep=";",
 #   thousands=",",
#)
#df


# In[ ]:





# In[13]:


#df = pd.read_csv(
#    "./results/compiled_results_for_RL/bpic2012/ready_to_use_scaled_adaptive_bpic2012.csv",
#    sep=";",
#)
#df


# In[ ]:




