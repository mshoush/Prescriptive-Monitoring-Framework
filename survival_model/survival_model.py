import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from sklearn.preprocessing import OneHotEncoder
from lifelines.utils import concordance_index
import os
from sys import argv
from tqdm import tqdm

# Constants
case_id_col = "case_id"
activity_col = 'activity'
resource_col = 'resource'
timestamp_col = 'timestamp'
label_col = 'label'
pos_label = 'deviant'
neg_label = 'regular'
treatment_cols = ['Treatment1', 'Treatment2', 'Treatment3', 'Treatment4']
pos_treatment = "Treatment"
neg_treatment = "Control"

# File Paths
dataset_name = argv[1]#"bpic2017"

# /prepared_data/{log_name}/train_encoded_{log_name}.csv

train_file_path = f"./prepared_data/{dataset_name}/train_encoded_{dataset_name}.csv"# argv[2] # "./../train_encoded_bpic2017.csv"
test_file_path = f"./prepared_data/{dataset_name}/test_encoded_{dataset_name}.csv" # argv[3] # "./../test_encoded_bpic2017.csv"
valid_file_path = f"./prepared_data/{dataset_name}/valid_encoded_{dataset_name}.csv"  # argv[4] # "./../val_encoded_bpic2017.csv"
results_dir = f"./results/survival/{dataset_name}" #argv[5] # "./../results/"

# Load Data
train_encoded = pd.read_csv(train_file_path, sep=';')
test_encoded = pd.read_csv(test_file_path, sep=';')
valid_encoded = pd.read_csv(valid_file_path, sep=';')

# Standardize Data
def standardize_data(df):
    df['event'] = 1
    df["event"] = df["event"].astype("bool")
    df["time_to_last_event_days"] = pd.to_numeric(df["time_to_last_event_days"])
    float64_cols = list(df.select_dtypes(include='float64'))
    df[float64_cols] = df[float64_cols].astype('float32')
    return df

print("Standardizing Data...")
train_encoded = standardize_data(train_encoded)
test_encoded = standardize_data(test_encoded)
valid_encoded = standardize_data(valid_encoded)

# Split Data
def split_data(df, ratio=0.4):
    train_data, calib_data = train_test_split(df, test_size=ratio, random_state=42, shuffle=True, stratify=df['Outcome'])
    return train_data, calib_data
print("Splitting Data...")
train_encoded, calib_encoded = split_data(train_encoded)

# Create Survival Model
def survival_model(X_train, y_train, time_col, ev_col):
    train_data = pd.concat([X_train, y_train], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_data, duration_col=time_col, event_col=ev_col, show_progress=True)
    return cph
print("Creating Survival Model...")
model = survival_model(train_encoded.drop('time_to_last_event_days', axis=1), train_encoded['time_to_last_event_days'], "time_to_last_event_days", "event")

# Predictions
def predict_chunks(model, X_data, chunk_size=10):
    chunks = [X_data[i:i + chunk_size] for i in range(0, len(X_data), chunk_size)]
    predictions = []
    for chunk in tqdm(chunks):
        chunk_predictions = model.predict_expectation(chunk)
        predictions.append(chunk_predictions)
    return np.concatenate(predictions)

# Calculate Quantiles
def calculate_q_yhat_naive(preds_cal, y_cal, alpha):
    N = len(y_cal)
    q_yhat = np.quantile(np.abs(y_cal - preds_cal), np.ceil((N + 1) * (1 - alpha)) / N)
    return q_yhat

print("Predict and Calculate Quantiles...")
alpha_values = np.round(np.arange(0.1, 1.0, 0.1), 1)
preds_cal = predict_chunks(model, calib_encoded.drop('time_to_last_event_days', axis=1))

qhat_naive = {a: calculate_q_yhat_naive(preds_cal, calib_encoded['time_to_last_event_days'], a) for a in alpha_values}

# Calculate Coverage
def calculate_coverage_naive(lower_bound, upper_bound, y_data, alpha):
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    y_data = np.array(y_data)
    out_of_bound = np.sum((y_data < lower_bound) | (y_data > upper_bound))
    return 1 - out_of_bound / len(y_data), lower_bound, upper_bound


print("Predicting Test and Validation Sets...")
# Predictions for Test and Validation Sets
preds_test = predict_chunks(model, test_encoded.drop('time_to_last_event_days', axis=1))
preds_valid = predict_chunks(model, valid_encoded.drop('time_to_last_event_days', axis=1))

# Calculate Intervals
pred_intervals_naive_test = {alpha: calculate_coverage_naive(preds_test - qhat, preds_test + qhat, test_encoded['time_to_last_event_days'], alpha) for alpha, qhat in qhat_naive.items()}
pred_intervals_naive_valid = {alpha: calculate_coverage_naive(preds_valid - qhat, preds_valid + qhat, valid_encoded['time_to_last_event_days'], alpha) for alpha, qhat in qhat_naive.items()}

# Add columns to DataFrames
def add_columns_to_df(df, pred_intervals):
    for alpha, (coverage, lower_bound, upper_bound) in pred_intervals.items():
        df[f"lower_bound_{alpha}"] = lower_bound
        df[f"upper_bound_{alpha}"] = upper_bound
        df[f"coverage_{alpha}"] = coverage


print("Adding Columns to DataFrames...")
# Add columns to Test and Validation DataFrames
add_columns_to_df(test_encoded, pred_intervals_naive_test)
add_columns_to_df(valid_encoded, pred_intervals_naive_valid)

# Save DataFrames to Results Directory
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


print("Saving DataFrames to Results Directory...")
# dataset_name = argv[1]#"bpic2017"
test_encoded.to_csv(os.path.join(results_dir, "survival_test_%s.csv"%dataset_name), sep=';', index=False)
valid_encoded.to_csv(os.path.join(results_dir, "survival_valid_%s.csv"%dataset_name), sep=';', index=False)
