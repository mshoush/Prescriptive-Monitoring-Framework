from causal_estimators.base import BaseEstimator, BaseIteEstimator
from causal_estimators.metalearners import *
from causal_estimators.forest_estimators import *
import pandas as pd
import numpy as np
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv2D, Conv1D
from tensorflow.keras.layers import BatchNormalization, Reshape
from tensorflow.keras.optimizers.schedules import ExponentialDecay
tf.keras.backend.clear_session()


class wtt_estimator:
    def __init__(self, args):
        self.args = args
        self.estimator_name = eval(args.estimator)
        self.propensity_model = args.propensity_model
        self.outcome_model = eval(args.outcome_model)
        # self.conf_thresh = args.conf_thresh

    def initialize_model(self):
        print('initializing model')
        self.estimator = self.estimator_name(outcome_models=self.outcome_model())

    def initialize_forest(self):
        print('initializing model')
        self.estimator = self.estimator_name()

    def fit_forest(self, X,T,Y):
        print('now fitting')
        self.estimator.fit(X, T, Y)

    def fit_estimator(self, X,T,Y):
        print('now fitting')
        self.estimator.fit(X, T, Y, conf_int_type='bootstrap')

    def get_te(self,X):
        print('estimating treatment effects')
        te = self.estimator.estimate_ite(w=X)
        return te

    def get_te_withCI(self,X):
        print('estimating confidence intervals')
        # te_lower, te_upper = self.estimator.effect_interval(X)
        self.te_lower, self.te_upper = self.estimator.estimate_CI(X)
        return self.te_lower, self.te_upper

    def get_CI_length(self):
        print('calculating interval length')
        self.interval = self.te_upper - self.te_lower
        return self.interval

    def save_results(self, df):
        self.results = df
        ## save to disk: implement later


    # def get_policy(self,te):
    #     value_est = te
    #     return value_est
    #
    # def get_policy_lowerTE(self,te_lower):
    #     value_est = te_lower
    #     return value_est
    #
    # def get_policy_withCI(self,te, interval):
    #     value_est = te/interval
    #     return value_est

    def calculate_cost(self, x):
        # self.te/self.interval * (B - C)
        B = 1
        C = B * self.ratio
        return x['treatment_effects']/x['interval'] * (B - C)
        # return costs[int(x['prediction']), int(x['actual'])](x)


    def evaluate_model_cost(self,args):
        conf_threshold = args['conf_threshold']
        dt_preds = self.results

        # trigger alarms according to conf_threshold
        dt_final = pd.DataFrame()
        unprocessed_case_ids = set(dt_preds.case_id.unique())
        for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
            tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
            tmp = tmp[tmp.treatment_effects >= conf_threshold]
            tmp["prediction"] = 1
            dt_final = pd.concat([dt_final, tmp], axis=0)
            unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == 1)]
        tmp["prediction"] = 0
        dt_final = pd.concat([dt_final, tmp], axis=0)

        case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
        case_lengths.columns = ["case_id", "case_length"]
        dt_final = dt_final.merge(case_lengths)

        cost = dt_final.apply(self.calculate_cost, axis=1).sum()

        return {'loss': cost, 'status': STATUS_OK, 'model': dt_final}

    def find_opt_thresh(self):
        print('Optimizing parameters...')
        cb_ratio = [5, 2, 1, 0.5, 0.3, 0.2, 0.1]
        # cost_weights = [(1, 1), (2, 1), (3, 1), (5, 1), (10, 1), (20, 1)]
        # c_postpone_weight = 0
        for ratio in cb_ratio:

            self.ratio = ratio
            space = {'conf_threshold': hp.uniform("conf_threshold", -1, 1)}
            trials = Trials()
            best = fmin(self.evaluate_model_cost, space, algo=tpe.suggest, max_evals=50, trials=trials)

            best_params = hyperopt.space_eval(space, best)
            print(best_params)

        return  best_params


        # for c_miss_weight, c_action_weight in cost_weights:
        #     c_miss = c_miss_weight / (c_miss_weight + c_action_weight)
        #     c_action = c_action_weight / (c_miss_weight + c_action_weight)
        #
        #     costs = np.matrix([[lambda x: 0,
        #                         lambda x: c_miss],
        #                        [lambda x: c_action + c_postpone_weight * (x['prefix_nr'] - 1) / x['case_length'],
        #                         lambda x: c_action + c_postpone_weight * (x['prefix_nr'] - 1) / x['case_length'] +
        #                                   (x['prefix_nr'] - 1) / x['case_length'] * c_miss
        #                         ]])


    def make_lstm_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(None, 1), return_sequences=True))  #
        model.add(LeakyReLU(alpha=0.05))
        # model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(LSTM(32, return_sequences=True))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.2))

        model.add(LSTM(32, return_sequences=True))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))

        model.add(LSTM(50))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))

        model.add(Dense(1, activation='sigmoid'))
        lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
        opt = Adam(learning_rate=lr_schedule)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        # model.summary()
        return model
