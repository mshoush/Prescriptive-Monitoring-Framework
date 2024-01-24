import os
import pandas as pd
from causallift import CausalLift
import sys
from sys import argv


def read_encoded_data(log_name):
    return (
        pd.read_csv(f"./prepared_data/{log_name}/train_encoded_{log_name}.csv", sep=";"),
        pd.read_csv(f"./prepared_data/{log_name}/test_encoded_{log_name}.csv", sep=";"),
        pd.read_csv(f"./prepared_data/{log_name}/valid_encoded_{log_name}.csv", sep=";")
    )

def create_results_directory(log_name):
    results_dir = f"./results/causal/{log_name}/"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def estimate_propensity_scores(train_data, test_data, valid_data, treatment_col):
    print(f"{treatment_col}\nRenaming treatment column...")
    train_data = train_data.rename(columns={treatment_col: 'Treatment'})
    test_data = test_data.rename(columns={treatment_col: 'Treatment'})
    valid_data = valid_data.rename(columns={treatment_col: 'Treatment'})

    print('\n[Estimate propensity scores for Inverse Probability Weighting.]')
    cl_train_test = CausalLift(
        train_data, test_data, enable_ipw=True, verbose=2,
        uplift_model_params={'cv': 3, 'estimator': 'xgboost.XGBClassifier', 'n_jobs': -1,
                             'param_grid': {'base_score': [0.5], 'booster': ['gbtree'],
                                            'colsample_bylevel': [1], 'colsample_bytree': [1], 'gamma': [0],
                                            'learning_rate': [0.1], 'max_delta_step': [0], 'max_depth': [3],
                                            'min_child_weight': [1], 'missing': [1], 'n_estimators': [100],
                                            'n_jobs': [-1], 'nthread': [None], 'objective': ['binary:logistic'],
                                            'random_state': [0], 'reg_alpha': [0], 'reg_lambda': [1],
                                            'scale_pos_weight': [1], 'subsample': [1], 'verbose': [0]},
                             'return_train_score': False, 'scoring': None,
                             'search_cv': 'sklearn.model_selection.GridSearchCV'},

        propensity_model_params={'cv': 3, 'estimator': 'sklearn.linear_model.LogisticRegression', 'n_jobs': -1,
                                 'param_grid': {'C': [0.1, 1, 10], 'class_weight': [None], 'dual': [False],
                                                'fit_intercept': [True], 'intercept_scaling': [1],
                                                'max_iter': [1000], 'multi_class': ['ovr'], 'n_jobs': [1],
                                                'penalty': ['l1', 'l2'], 'random_state': [0],
                                                'solver': ['liblinear'], 'tol': [0.0001], 'warm_start': [False]},
                                 'return_train_score': False, 'scoring': None,
                                 'search_cv': 'sklearn.model_selection.GridSearchCV'})

    cl_train_val = CausalLift(
        train_data, valid_data, enable_ipw=True, verbose=2,
        uplift_model_params={'cv': 3, 'estimator': 'xgboost.XGBClassifier', 'n_jobs': -1,
                             'param_grid': {'base_score': [0.5], 'booster': ['gbtree'],
                                            'colsample_bylevel': [1], 'colsample_bytree': [1], 'gamma': [0],
                                            'learning_rate': [0.1], 'max_delta_step': [0], 'max_depth': [3],
                                            'min_child_weight': [1], 'missing': [1], 'n_estimators': [100],
                                            'n_jobs': [-1], 'nthread': [None], 'objective': ['binary:logistic'],
                                            'random_state': [0], 'reg_alpha': [0], 'reg_lambda': [1],
                                            'scale_pos_weight': [1], 'subsample': [1], 'verbose': [0]},
                             'return_train_score': False, 'scoring': None,
                             'search_cv': 'sklearn.model_selection.GridSearchCV'},

        propensity_model_params={'cv': 3, 'estimator': 'sklearn.linear_model.LogisticRegression', 'n_jobs': -1,
                                 'param_grid': {'C': [0.1, 1, 10], 'class_weight': [None], 'dual': [False],
                                                'fit_intercept': [True], 'intercept_scaling': [1],
                                                'max_iter': [1000], 'multi_class': ['ovr'], 'n_jobs': [1],
                                                'penalty': ['l1', 'l2'], 'random_state': [0],
                                                'solver': ['liblinear'], 'tol': [0.0001], 'warm_start': [False]},
                                 'return_train_score': False, 'scoring': None,
                                 'search_cv': 'sklearn.model_selection.GridSearchCV'})

    return cl_train_test, cl_train_val

def estimate_cate_and_effect(cl_train_test, cl_train_val, results_dir, dataset_name, treatment_col):
    print('\n[Create 2 models for treatment and untreatment and estimate CATE (Conditional Average Treatment Effects)]')
    train_df1, test_df = cl_train_test.estimate_cate_by_2_models()

    print('\n[Show CATE for train dataset]')
    # Save results
    train_df1.to_csv(os.path.join(results_dir, f'causalLift_train_df_CATE_{dataset_name}_{treatment_col}.csv'),
                     index=False, sep=';')

    print('\n[Show CATE for test dataset]')
    # Save results
    test_df.to_csv(os.path.join(results_dir, f'causalLift_test_df_CATE_{dataset_name}_{treatment_col}.csv'),
                   index=False, sep=';')

    train_df2, valid_df = cl_train_val.estimate_cate_by_2_models()

    # Save results
    train_df2.to_csv(os.path.join(results_dir, f'causalLift_train_df2_CATE_{dataset_name}_{treatment_col}.csv'),
                     index=False, sep=';')
    valid_df.to_csv(os.path.join(results_dir, f'causalLift_valid_df_CATE_{dataset_name}_{treatment_col}.csv'),
                    index=False, sep=';')

    print('\n[Estimate the effect of recommendation based on the uplift model]')
    estimated_effect_df_cl_train_test = cl_train_test.estimate_recommendation_impact()

    # Save results
    estimated_effect_df_cl_train_test.to_csv(
        os.path.join(results_dir, f'causalLift_estimated_effect_df_{dataset_name}_{treatment_col}.csv'),
        index=False, sep=';')

    print('\n[Estimate the effect of recommendation based on the uplift model]')
    estimated_effect_df_cl_train_valid = cl_train_val.estimate_recommendation_impact()

    # Save results
    estimated_effect_df_cl_train_valid.to_csv(
        os.path.join(results_dir, f'causalLift_estimated_effect_df_{dataset_name}_{treatment_col}.csv'),
        index=False, sep=';')

def main():
    log_name = argv[1]#"bpic2017"  # Set your log_name here
    train_data, test_data, valid_data = read_encoded_data(log_name)
    results_dir = create_results_directory(log_name)

    if log_name == "bpic2017":
        treatment_cols = ["Treatment1", "Treatment2", "Treatment3", "Treatment4"]
    else:
        treatment_cols = ["Treatment1"]

    for treatment_col in treatment_cols:
        estimate_cate_and_effect(*estimate_propensity_scores(train_data, test_data, valid_data, treatment_col),
                                 results_dir, log_name, treatment_col)

if __name__ == "__main__":
    main()
