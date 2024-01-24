import os
import pandas as pd
import numpy as np

class ConformalPrediction:
    def __init__(self, prediction_cal, prediction_test, y_true_cal, y_true_test, alpha):
        self.prediction_cal = prediction_cal
        self.prediction_test = prediction_test
        self.y_true_cal = y_true_cal
        self.y_true_test = y_true_test
        self.alpha = alpha

    def get_qhat_naive(self, alpha):
        N = self.prediction_cal.shape[0]
        scores = 1 - np.array([self.prediction_cal[i, self.y_true_cal[i]] for i in range(N)])
        q_yhat = np.quantile(scores, np.ceil((N + 1) * (1 - alpha)) / N)
        return q_yhat

    def get_pred_set_naive(self, q_yhat):
        softmax_outputs = self.prediction_test
        pred_sets = [[j for j in range(softmax_outputs.shape[1]) if softmax_outputs[i, j] >= 1 - q_yhat] for i in range(softmax_outputs.shape[0])]
        return pred_sets

    def get_ind(self, y_pred_i, y_test_i):
        return np.where(np.sort(y_pred_i)[::-1] == y_pred_i[y_test_i])[0][0]

    def get_qhat_adaptive(self, alpha):
        sums_density = [np.sum(np.sort(self.prediction_cal[i])[::-1][:self.get_ind(self.prediction_cal[i], self.y_true_cal[i]) + 1]) for i in range(self.y_true_cal.shape[0])]
        N = self.y_true_cal.shape[0]
        q_yhat = np.quantile(sums_density, np.ceil((N + 1) * (1 - alpha)) / N)
        return q_yhat

    def get_pred_set_adaptive(self, q_yhat):
        conf_sets = [[np.where(self.prediction_test == np.sort(self.prediction_test[i])[::-1][j])[1][0] for j in range(np.where(np.cumsum(np.sort(self.prediction_test[i])[::-1]) > q_yhat)[0][0] + 1)] for i in range(self.prediction_test.shape[0])]
        return conf_sets

    def get_qhat_class_balance(self, alpha):
        d_alpha = {real_class: [1 - self.prediction_cal[i, real_class] for i in range(self.y_true_cal.shape[0]) if self.y_true_cal[i] == real_class] for real_class in set(self.y_true_cal)}
        d_alpha_q_yhats = {k: np.quantile(v, np.ceil((len(v) + 1) * (1 - alpha)) / len(v)) for k, v in d_alpha.items()}
        return d_alpha_q_yhats

    def get_pred_set_class_balance(self, d_alpha_q_yhats):
        conf_sets = [[j for j in range(self.prediction_test.shape[1]) if 1 - self.prediction_test[i, j] < d_alpha_q_yhats[j]] for i in range(self.prediction_test.shape[0])]
        return conf_sets

    def get_pred_set_size(self, alpha, pred_sets):
        my_dict_alpha = {a: {str(key): val for key, val in zip(pd.Series(pred_sets[a]).value_counts().index[:].to_list(), pd.Series(pred_sets[a]).value_counts().to_list())} for a in alpha}
        return my_dict_alpha


def get_df_with_pred_sets(df, alpha, pred_set):
    df = df.copy(deep=True)
    for a in alpha:
        df[f"alpha_{str(a)}"] = pred_set[a]
    return df


def main(dataset_name, pred_test, pred_train, pred_cal, results_causal):
    y_test = pred_test['actual'].to_numpy()
    prediction_test = np.column_stack((pred_test['predicted_proba_0'], pred_test['predicted_proba_1']))
    y_train = pred_train['actual'].to_numpy()
    prediction_train = np.column_stack((pred_train['predicted_proba_0'], pred_train['predicted_proba_1']))
    y_cal = pred_cal['actual'].to_numpy()
    prediction_cal = np.column_stack((pred_cal['predicted_proba_0'], pred_cal['predicted_proba_1']))

    alpha = np.round(np.arange(0.1, 1.0, 0.1), 1)

    cp = ConformalPrediction(prediction_cal=prediction_cal,
                              prediction_test=prediction_test,
                              y_true_cal=y_cal,
                              y_true_test=y_test,
                              alpha=alpha)

    print("Start Conformal Prediction")
    
    qhat_class_balance = {a: cp.get_qhat_class_balance(a) for a in alpha}
    pred_sets_class_balance = {alpha: cp.get_pred_set_class_balance(qhat) for alpha, qhat in qhat_class_balance.items()}
    pred_sets_size_class_balance = cp.get_pred_set_size(alpha, pred_sets_class_balance)

    df_test_result = pred_test.copy(deep=True)
    df_class = get_df_with_pred_sets(df_test_result, list(pred_sets_class_balance.keys()), pred_sets_class_balance)
    return df_class

from sys import argv

if __name__ == "__main__":
    dataset = argv[1]
    datasets = [dataset]
    # ./../results/predictive/bpic2017/
    results_pred = argv[2] #"/home/centos/phd/4thyear/RL-prescriptive-monitoring/results/predictive/%s" % dataset
    results_conformal = argv[3] #"/home/centos/phd/4thyear/RL-prescriptive-monitoring/results/conformal/%s" % dataset
    results_causal = argv[4] #"/home/centos/phd/4thyear/RL-prescriptive-monitoring/results/causal/%s" % dataset

    if not os.path.exists(os.path.join(results_conformal)):
        os.makedirs(os.path.join(results_conformal))


    # Function to check if a file exists
    def file_exists(file_path):
        return os.path.exists(file_path)


    def read_data(data_type, dataset_name, results_dir):
        data_path = os.path.join(results_dir, f"preds_{data_type}_{dataset_name}.csv")
        if file_exists(data_path):
            print(f"Reading existing {data_type} encoded data...")
            return pd.read_csv(data_path, sep=";")
        else:
            print(f"{data_type} encoded data does not exist. Applying encode_data function...")

    for dataset in datasets:
        print("dataset: ", dataset)

        pred_test = read_data("test", dataset, results_pred)
        pred_train = read_data("train", dataset, results_pred)
        pred_cal = pred_train.sample(frac=0.4, random_state=42)
        pred_train = pred_train.drop(pred_cal.index)
        pred_val = read_data("valid", dataset, results_pred)

        #pred_test = pd.read_csv(os.path.join(results_pred, "preds_test_%s.csv" % dataset), sep=';')
        #pred_train = pd.read_csv(os.path.join(results_pred, "preds_train_%s.csv" % dataset), sep=';')
        #pred_cal = pred_train.sample(frac=0.4, random_state=42)
        #pred_train = pred_train.drop(pred_cal.index)
        #pred_val = pd.read_csv(os.path.join(results_pred, "preds_val_%s.csv" % dataset), sep=';')

        df_class_test = main(dataset, pred_test, pred_train, pred_cal, results_causal)
        df_class_valid = main(dataset, pred_val, pred_train, pred_cal, results_causal)

        # Save results
        df_class_test.to_csv(os.path.join(results_conformal, "conformal_test_%s.csv" % dataset), sep=";", index=False)
        df_class_valid.to_csv(os.path.join(results_conformal, "conformal_valid_%s.csv" % dataset), sep=";", index=False)






