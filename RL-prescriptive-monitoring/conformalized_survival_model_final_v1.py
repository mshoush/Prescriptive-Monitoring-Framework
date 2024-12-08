
import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder

import os
from sys import argv

dataset_name = argv[1]

train_data = pd.read_pickle("./results/predictive/%s/train_.pkl"%dataset_name)
test_data = pd.read_pickle("./results/predictive/%s/test_.pkl"%dataset_name)
cal_data = pd.read_pickle("./results/predictive/%s/valid_.pkl"%dataset_name)




#train_data = pd.read_pickle("./results/predicitve/bpic2012/train_.pkl")
#est_data = pd.read_pickle("./results/predicitve/bpic2012/test_.pkl")
#cal_data = pd.read_pickle("./results/predicitve/bpic2012/valid_.pkl")

print("Read encoded data...")
df_agg = pd.read_csv(
    "./results/predictive/%s/dt_transformed_agg_%s.csv"%(dataset_name, dataset_name),
    low_memory=False,
    sep=";",
)
df_static = pd.read_csv(
    "./results/predictive/%s/dt_transformed_static_%s.csv"%(dataset_name, dataset_name),
    low_memory=False,
    sep=";",
)

static_agg_df = pd.concat([df_static, df_agg], axis=1)
cat_feat_idx = np.where(static_agg_df.dtypes != float)[0]

#  rename columns
train_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]
#print(train_data.columns)
test_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]
cal_data.columns = list(static_agg_df.columns) + ["Outcome"]  # + ["Treatment"]


train_data["time_to_event_m"] = pd.to_numeric(train_data["time_to_event_m"])
test_data["time_to_event_m"] = pd.to_numeric(test_data["time_to_event_m"])
cal_data["time_to_event_m"] = pd.to_numeric(cal_data["time_to_event_m"])



cal_df = cal_data.copy()
test_df = test_data.copy()
train_df = train_data.copy()

#del cal_data
#del test_data
#del train_data

# Select columns with 'float64' dtype  
float64_cols = list(cal_df.select_dtypes(include='float64'))
cal_df[float64_cols] = cal_df[float64_cols].astype('float32')

# Select columns with 'float64' dtype  
float64_cols = list(test_df.select_dtypes(include='float64'))
test_df[float64_cols] = test_df[float64_cols].astype('float32')

# Select columns with 'float64' dtype  
float64_cols = list(train_df.select_dtypes(include='float64'))
train_df[float64_cols] = train_df[float64_cols].astype('float32')



dfs = [cal_df, test_df, train_df]
for df in dfs:

    for i in cat_feat_idx:
        col = df.columns[i]

        if col == "event":
            continue
        te = TargetEncoder()
        df[col] = te.fit_transform(df[col], df.time_to_event_m)


# Add event column that indicates all observations are uncencored
# All observation are complete
train_df["event"] = 1
train_df["event"] = train_df["event"].astype("bool")

test_df["event"] = 1
test_df["event"] = test_df["event"].astype("bool")

cal_df["event"] = 1
cal_df["event"] = cal_df["event"].astype("bool")

# Remove theses columns becusal they are not working withthe Cox model


try:
    train_df = train_df[train_df.columns.difference(["timesincelastevent_min", "timesincecasestart_min", "event_nr_min", 'Accepted', 'CreditScore_min', 'FirstWithdrawalAmount_min', 'MonthlyCost_min', 'NumberOfTerms_min', 'OfferedAmount_min', 'Selected'])]
    test_df = test_df[test_df.columns.difference(["timesincelastevent_min", "timesincecasestart_min", "event_nr_min", 'Accepted', 'CreditScore_min', 'FirstWithdrawalAmount_min', 'MonthlyCost_min', 'NumberOfTerms_min', 'OfferedAmount_min', 'Selected'])]
    cal_df = cal_df[cal_df.columns.difference(["timesincelastevent_min", "timesincecasestart_min", "event_nr_min", 'Accepted', 'CreditScore_min', 'FirstWithdrawalAmount_min', 'MonthlyCost_min', 'NumberOfTerms_min', 'OfferedAmount_min', 'Selected'] ) ]
except:
    train_df = train_df[train_df.columns.difference(["timesincelastevent_min", "timesincecasestart_min", "event_nr_min"])]
    test_df = test_df[test_df.columns.difference(["timesincelastevent_min", "timesincecasestart_min", "event_nr_min"])]
    cal_df = cal_df[cal_df.columns.difference(["timesincelastevent_min", "timesincecasestart_min", "event_nr_min"]   )]



# In[3]:



from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import gc
gc.enable()
# Check if the garbage collector is enabled
if gc.isenabled():
    print("Garbage collector is enabled")
else:
    print("Garbage collector is disabled")


y_test = test_df["time_to_event_m"]
X_test = test_df.drop(["time_to_event_m"], axis=1)

y_train = train_df["time_to_event_m"]
X_train = train_df.drop(["time_to_event_m"], axis=1)

y_cal = cal_df["time_to_event_m"]
X_cal = cal_df.drop(["time_to_event_m"], axis=1)

X_cal1, X_cal2, y_cal1, y_cal2 = train_test_split(
    X_cal, y_cal, test_size=0.5, random_state=22
)



# In[4]:


time_col = "time_to_event_m"
ev_col = "event"


def survival_model(X_train, y_train, time_col, ev_col):
    train_data = pd.concat([X_train, y_train], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        train_data, duration_col=time_col, event_col=ev_col, show_progress=True,
    )

    return cph


# # Naive method

# In[5]:


model = survival_model(X_train, y_train, time_col, ev_col)
#print(X_cal)
#print(X_cal.dtypes)
# Manually trigger garbage collection
# Select columns with 'float64' dtype  
#print(X_cal.dtypes)
#float64_cols = list(X_cal.select_dtypes(include='float64'))
#X_cal[float64_cols] = X_cal[float64_cols].astype('float32')

#print(X_cal.dtypes)


# Split the DataFrame into smaller chunks
chunk_size = 100000  # Adjust the chunk size as per your system's memory capacity
chunks = [X_cal[i:i+chunk_size] for i in range(0, len(X_cal), chunk_size)]

# Initialize an empty list to store the predicted values
predictions = []

# Iterate over the chunks and predict on each chunk
print("1")
i =0
for chunk in chunks:
    print(f"chunk: {i}...")
    i+=1
    # Predict on the chunk
    chunk_predictions = model.predict_expectation(chunk)
    # Append the predictions to the list
    predictions.append(chunk_predictions)


# Concatenate the predictions from all chunks
preds_cal = np.concatenate(predictions)
#preds_cal = np.asarray(predictions, dtype=np.float32)
#preds_cal = predictions
print(preds_cal)

gc.collect()
print("Before...")
#preds_cal = model.predict_expectation(X_cal)
print("After...")

# In[6]:


def calculate_q_yhat_naive(preds_cal, y_cal, alpha):
    print(f"\nAlpha: {alpha}")

    N = len(y_cal)
    q_yhat = np.quantile(np.abs(y_cal - preds_cal), np.ceil((N + 1) * (1 - alpha)) / N)
    print(f"qhat: {q_yhat}")

    return q_yhat


# conformal prediction object
alpha = np.round(np.arange(0.1, 1.0, 0.1), 1)

# Naive
qhat_naive = {a: calculate_q_yhat_naive(preds_cal, y_cal, a) for a in alpha}
qhat_naive


# In[7]:

print("2")
# Split the DataFrame into smaller chunks
chunk_size = 100000  # Adjust the chunk size as per your system's memory capacity
chunks = [X_test[i:i+chunk_size] for i in range(0, len(X_test), chunk_size)]

# Initialize an empty list to store the predicted values
predictions = []

# Iterate over the chunks and predict on each chunk
i =0
for chunk in chunks:
    print(f"chunk: {i}...")
    i+=1
    # Predict on the chunk
    chunk_predictions = model.predict_expectation(chunk)
    # Append the predictions to the list
    predictions.append(chunk_predictions)


# Concatenate the predictions from all chunks
preds_test = np.concatenate(predictions)



#preds_test = np.array(model.predict_expectation(X_test))
#preds_test


# In[8]:


def calculate_coverage_naive(lower_bound, upper_bound, y_test, alpha):

    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    y_test = np.array(y_test)
    out_of_bound = 0
    N = len(y_test)

    for i in range(N):
        if y_test[i] < lower_bound[i] or y_test[i] > upper_bound[i]:
            out_of_bound += 1

    #     print(
    #         f"Alpha is: {alpha}, with Coverage of {1 - out_of_bound / N}, Lower bound: {lower_bound}, and Upper bound: {upper_bound}"
    #     )
    return 1 - out_of_bound / N, lower_bound, upper_bound


# lower_bound = preds_test - qhat
# upper_bound = preds_test + qhat


pred_intervals_naive = {
    alpha: calculate_coverage_naive(preds_test - qhat, preds_test + qhat, y_test, alpha)
    for alpha, qhat in qhat_naive.items()
}
pred_intervals_naive


# # Adaptative intervals
# 
# Conformalized residual fitting

# In[9]:

# Split the DataFrame into smaller chunks
print("3")
chunk_size = 1000  # Adjust the chunk size as per your system's memory capacity
chunks = [X_cal1[i:i+chunk_size] for i in range(0, len(X_cal1), chunk_size)]

# Initialize an empty list to store the predicted values
predictions = []

# Iterate over the chunks and predict on each chunk
i =0
for chunk in chunks:
    print(f"chunk: {i}...")
    i+=1
    # Predict on the chunk
    chunk_predictions = model.predict_expectation(chunk)
    # Append the predictions to the list
    predictions.append(chunk_predictions)


# Concatenate the predictions from all chunks
preds_X_cal1 = np.concatenate(predictions)



# calculate residuals
r_y = np.abs(y_cal1 - preds_X_cal1)
r_y = pd.DataFrame(r_y,)
r_y.columns = ["time_to_event_m"]
r_y["time_to_event_m"] = pd.to_numeric(r_y["time_to_event_m"])


# fit model residual on residuals
model_r = survival_model(X_cal1, r_y, time_col, ev_col)


# In[10]:


# check for coverage without ICP
#preds_test = model.predict_expectation(X_test)
print("4")
# Split the DataFrame into smaller chunks
chunk_size = 1000  # Adjust the chunk size as per your system's memory capacity
chunks = [X_test[i:i+chunk_size] for i in range(0, len(X_test), chunk_size)]

# Initialize an empty list to store the predicted values
predictions = []

# Iterate over the chunks and predict on each chunk
i =0
for chunk in chunks:
    print(f"chunk: {i}...")
    i+=1
    # Predict on the chunk
    chunk_predictions = model_r.predict_expectation(chunk)
    # Append the predictions to the list
    predictions.append(chunk_predictions)


# Concatenate the predictions from all chunks
preds_test_r = np.concatenate(predictions)


#preds_test_r = model_r.predict_expectation(X_test)


# In[11]:


# Without conformal
lower_bound = preds_test - preds_test_r
upper_bound = preds_test + preds_test_r


pred_intervals_adaptive_1 = {
    alpha: calculate_coverage_naive(
        np.array(lower_bound), np.array(upper_bound), y_test, alpha
    )
    for alpha, qhat in qhat_naive.items()
}
pred_intervals_adaptive_1


# In[12]:


# With Conformal
# calculate q_yhat
print("5")
# Split the DataFrame into smaller chunks
chunk_size = 1000  # Adjust the chunk size as per your system's memory capacity
chunks = [X_cal2[i:i+chunk_size] for i in range(0, len(X_cal2), chunk_size)]

# Initialize an empty list to store the predicted values
predictions1 = []
predictions2 = []

# Iterate over the chunks and predict on each chunk
i =0
for chunk in chunks:
    print(f"chunk: {i}...")
    i+=1
    # Predict on the chunk
    chunk_predictions1 = model.predict_expectation(chunk)
    chunk_predictions2 = model_r.predict_expectation(chunk)
    # Append the predictions to the list
    predictions1.append(chunk_predictions1)
    predictions2.append(chunk_predictions2)


# Concatenate the predictions from all chunks
#preds_test_r = np.concatenate(predictions)


preds_cal2 = np.concatenate(predictions1) #model.predict_expectation(X_cal2)
preds_cal2_r = np.concatenate(predictions2) #model_r.predict_expectation(X_cal2)
N = len(y_cal2)

def calculate_q_yhat_adaptive(preds_cal2, preds_cal2_r, y_cal2, a):
    
    q_yhat = np.quantile(np.abs(y_cal2 - preds_cal2) / preds_cal2_r, np.ceil((N + 1) * (1 - a)) / N)
    return q_yhat




# conformal prediction object
alpha = np.round(np.arange(0.1, 1.0, 0.1), 1)

# Naive
qhat_adaptive = {
    a: calculate_q_yhat_adaptive(preds_cal2, preds_cal2_r, y_cal2, a) for a in alpha
}  # {alhp: q_hat}


print(qhat_adaptive)


# In[13]:


pred_intervals_adaptive_2 = {
    alpha: calculate_coverage_naive(
        preds_test - qhat * preds_test_r,
        preds_test + qhat * preds_test_r,
        y_test,
        alpha,
    )
    for alpha, qhat in qhat_adaptive.items()
}
pred_intervals_adaptive_2


# 
# # Conformalized Quantile Regression
# 
# Solution for the foregoing problem: Quantile Regression

# In[14]:


alpha = np.round(np.arange(0.1, 1.0, 0.1), 1)


def calibrate_qyhat(y_true, lower_bound, upper_bound, alpha):

    N = len(y_true)
    s = np.amax([lower_bound - y_true, y_true - upper_bound], axis=0)
    q_yhat = np.quantile(s, np.ceil((N + 1) * (1 - alpha)) / N)

    return q_yhat

print("6")
# Split the DataFrame into smaller chunks
chunk_size = 1000  # Adjust the chunk size as per your system's memory capacity
chunks = [X_cal[i:i+chunk_size] for i in range(0, len(X_cal), chunk_size)]

# Initialize an empty list to store the predicted values
predictions1 = []
predictions2 = []

# Iterate over the chunks and predict on each chunk
i =0
for chunk in chunks:
    print(f"chunk: {i}...")
    i+=1
    # Predict on the chunk
    chunk_predictions1 = model.predict_expectation(chunk)
    chunk_predictions2 = model_r.predict_expectation(chunk)
    # Append the predictions to the list
    predictions1.append(chunk_predictions1)
    predictions2.append(chunk_predictions2)


# check for coverage without ICP
preds_cal =  np.concatenate(predictions1) #model.predict_expectation(X_cal)
preds_cal_r = np.concatenate(predictions2) #model_r.predict_expectation(X_cal)
lower_bound = preds_cal - preds_cal_r
upper_bound = preds_cal + preds_cal_r

# Adaptive
qhat_adaptive_QR = {
    a: calibrate_qyhat(y_cal, lower_bound, upper_bound, a) for a in alpha
}  # {alhp: q_hat}


qhat_adaptive_QR


# In[15]:


print("7")
# Split the DataFrame into smaller chunks
chunk_size = 1000  # Adjust the chunk size as per your system's memory capacity
chunks = [X_test[i:i+chunk_size] for i in range(0, len(X_test), chunk_size)]

# Initialize an empty list to store the predicted values
predictions = []

# Iterate over the chunks and predict on each chunk
i =0
for chunk in chunks:
    print(f"chunk: {i}...")
    i+=1
    # Predict on the chunk
    chunk_predictions = model.predict_expectation(chunk)
    # Append the predictions to the list
    predictions.append(chunk_predictions)


# Concatenate the predictions from all chunks
forecast = np.concatenate(predictions)


#forecast = model.predict_expectation(X_test)


pred_intervals_adaptive_3 = {
    alpha: calculate_coverage_naive(
        np.array(forecast - qhat), np.array(forecast + qhat), y_test, alpha,
    )
    for alpha, qhat in qhat_adaptive_QR.items()
}
pred_intervals_adaptive_3


# In[16]:


# Naive method
pred_intervals_naive


# In[17]:


# Adaptive without conformal
pred_intervals_adaptive_1


# In[18]:


# Adaptative intervals - Conformalized residual fitting

pred_intervals_adaptive_2


# In[19]:


# Adaptative intervals -  Conformalized Quantile Regression

pred_intervals_adaptive_3


# In[34]:


dt_test_prefixes = pd.read_csv(
    "./results/predictive/%s/dt_test_prefixes_%s.csv"%(dataset_name, dataset_name), sep=";"
)
dt_test_prefixes.columns


# In[35]:


test_conformal_causal = pd.read_csv(
    "./results/conformal_causal/%s/test2_conformalizedTE_%s.csv"%(dataset_name, dataset_name), sep=";"
)
test_conformal_causal


# In[36]:


test_conformal_causal["prefix_nr"] = list(
    dt_test_prefixes.groupby("Case ID").first()["prefix_nr"]
)
test_conformal_causal["case_id"] = list(
    dt_test_prefixes.groupby("Case ID").first()["orig_case_id"]
)
test_conformal_causal["activity"] = list(
    dt_test_prefixes.groupby("Case ID").last()["Activity"]
)
try:
    test_conformal_causal["timestamp"] = list(dt_test_prefixes.groupby("Case ID").last()["start_time"])
except:
    test_conformal_causal["timestamp"] = list(dt_test_prefixes.groupby("Case ID").last()["time:timestamp"])
test_conformal_causal = test_conformal_causal.sort_values(by=["timestamp"]).reset_index(
    drop=True
)
test_conformal_causal


# In[ ]:





# In[21]:


test_data["prefix_nr"] = list(dt_test_prefixes.groupby("Case ID").first()["prefix_nr"])
test_data["case_id"] = list(dt_test_prefixes.groupby("Case ID").first()["orig_case_id"])
test_data["activity"] = list(dt_test_prefixes.groupby("Case ID").last()["Activity"])
try:
    test_data["timestamp"] = list(dt_test_prefixes.groupby("Case ID").last()["start_time"])
except:
    test_data["timestamp"] = list(dt_test_prefixes.groupby("Case ID").last()["time:timestamp"])
test_data = test_data.sort_values(by=["timestamp"]).reset_index(drop=True)


# In[22]:


test_data["predicted_time_to_event"] = preds_test


# In[23]:


# lower and upper bounds for Alpha = 0.1, it gives the highest coverage.
test_data["lower_time_to_event_adaptive"] = pred_intervals_adaptive_2[0.1][1]
test_data["upper_time_to_event_adaptive"] = pred_intervals_adaptive_2[0.1][2]


# In[24]:


# lower and upper bounds for Alpha = 0.1, it gives the highest coverage.
test_data["lower_time_to_event_adaptive_QR"] = pred_intervals_adaptive_3[0.1][1]
test_data["upper_time_to_event_adaptive_QR"] = pred_intervals_adaptive_3[0.1][2]


# In[25]:


# lower and upper bounds for Alpha = 0.1, it gives the highest coverage.
test_data["lower_time_to_event_naive"] = pred_intervals_naive[0.1][1]
test_data["upper_time_to_event_naive"] = pred_intervals_naive[0.1][2]


# In[26]:


test_data


# In[27]:


import os

results_surv = "./results/conformal_survival/%s/"%dataset_name
# create results directory
if not os.path.exists(os.path.join(results_surv)):
    os.makedirs(os.path.join(results_surv))

test_data.to_csv(
    os.path.join(results_surv, "test_data_survival_conformal_%s.csv" % dataset_name,),
    sep=";",
    index=False,
)


# In[37]:


import os

results_surv = "./results/conformal_causal/%s/"%dataset_name
# create results directory
if not os.path.exists(os.path.join(results_surv)):
    os.makedirs(os.path.join(results_surv))

test_conformal_causal.to_csv(
    os.path.join(results_surv, "test_data_causal_conformal_%s.csv" % dataset_name,),
    sep=";",
    index=False,
)


# In[ ]:




