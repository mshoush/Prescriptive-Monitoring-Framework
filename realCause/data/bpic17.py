import os
import pandas as pd
from utils import to_data_format, NUMPY, PANDAS_SINGLE, DATA_FOLDER #, DATA_NAME

# from utils import to_data_format 
# import utils.NUMPY
# import utils.PANDAS_SINGLE
# import utils.DATA_FOLDER
# import utils.DATA_NAME





BPIC17 = 'bpic17'
BPIC20 = 'bpic20'

print(DATA_FOLDER)

def load_bpic(DATA_NAME, data_format=NUMPY, dataroot=None, year=17):
    print(DATA_NAME)
    if DATA_NAME=="bpic17":
        DATA_NAME="bpic2017"
    elif DATA_NAME=="bpic12":
        DATA_NAME="bpic2012"
    """
    Load LaLonde dataset: RCT or combined RCT with observational control group
    Options for 2 x 6 = 12 different observational datasets and 2 RCT datasets

    :param rct_version: 'lalonde' for LaLonde (1986)'s original RCT data or 'dw' for Dehejia & Wahba (1999)'s RCT data
    :param obs_version: observational data to use for the control group
    :param rct: use RCT data for both the treatment AND control groups (no observational data)
    :param data_format: returned data format: 'torch' Tensors, 'pandas' DataFrame, or 'numpy' ndarrays
    :return: (covariates, treatment, outcome) tuple or Pandas DataFrame
    """
    if year==16:
        df = load_bpi16(dataroot=DATA_FOLDER)
        if data_format.lower() == PANDAS_SINGLE:
            return df
        else:
            w = df.drop(['CustomerID', 'treatment', 'outcome'], axis='columns')
            t = df['treatment']
            y = df['outcome']
            return to_data_format(data_format, w, t, y)

    if year==17:
        df = load_bpi17(DATA_NAME, dataroot=DATA_FOLDER + DATA_NAME + "/")
        if data_format.lower() == PANDAS_SINGLE:
            return df
        else:
            w = df.drop(['case_id', 'Treatment1', 'Outcome'], axis='columns')
            t = df['Treatment1']
            y = df['Outcome']
            return to_data_format(data_format, w, t, y)

    if year==12:
        df = load_bpi12(DATA_NAME, dataroot=DATA_FOLDER+DATA_NAME+"/")
        if data_format.lower() == PANDAS_SINGLE:
            return df
        else:
            w = df.drop(['case_id', 'Treatment1', 'Outcome'], axis='columns')
            t = df['Treatment1']
            y = df['Outcome']
            return to_data_format(data_format, w, t, y)

    if year==20:
        df = load_bpi20(dataroot=DATA_FOLDER)
        if data_format.lower() == PANDAS_SINGLE:
            return df
        else:
            w = df.drop(['case:id', 'treatment', 'duration'], axis='columns')
            t = df['treatment']
            y = df['duration']
            return to_data_format(data_format, w, t, y)

    if year==19:
        df = load_bpi19(dataroot=DATA_FOLDER)
        if data_format.lower() == PANDAS_SINGLE:
            return df
        else:
            w = df.drop(['Case ID', 'treatment', 'duration'], axis='columns')
            t = df['treatment']
            y = df['duration']
            return to_data_format(data_format, w, t, y)

    if year==22:
        df = load_bpi22(dataroot=DATA_FOLDER)
        # df["Case ID orig"] = df.index
        # df["Case ID"] = df["Case ID"].apply(lambda x: x.split("_")[1])

        grouped = df.groupby('Case ID')
        start_timestamps = grouped[['timesincecasestart', 'timesincefirstcase']].min().reset_index()
        start_timestamps = start_timestamps.sort_values('timesincefirstcase', ascending=True, kind='mergesort')
        train_ids = list(start_timestamps['Case ID'])[:int(0.8 * len(start_timestamps))]
        test_ids = list(start_timestamps['Case ID'])[-int(0.1 * len(start_timestamps)):]
        val_ids = list(set(start_timestamps['Case ID'])-(set(train_ids).union(set(test_ids))))
        # train_idx = [int(x) for x in train_ids]
        # test_idx = [int(x) for x in test_ids]
        # val_idx = [int(x) for x in val_ids]
        train_idx = df.index[df["Case ID"].isin(train_ids)].tolist()
        val_idx = df.index[df["Case ID"].isin(val_ids)].tolist()
        test_idx = df.index[df["Case ID"].isin(test_ids)].tolist()
        if data_format.lower() == PANDAS_SINGLE:
            return df
        else:
            w = df.drop(['treatment', 'outcome'], axis='columns')
            t = df['treatment']
            y = df['outcome']
        w,t,y = to_data_format(data_format, w, t, y)
        return w,t,y, train_idx, val_idx, test_idx

    # if year==12:
    #     df = load_bpi12(dataroot=DATA_FOLDER)
    #     # df["Case ID orig"] = df.index
    #     # df["Case ID"] = df["Case ID"].apply(lambda x: x.split("_")[1])

    #     grouped = df.groupby('Case ID')
    #     start_timestamps = grouped[['timesincecasestart', 'timesincefirstcase']].min().reset_index()
    #     start_timestamps = start_timestamps.sort_values('timesincefirstcase', ascending=True, kind='mergesort')
    #     train_ids = list(start_timestamps['Case ID'])[:int(0.8 * len(start_timestamps))]
    #     test_ids = list(start_timestamps['Case ID'])[-int(0.1 * len(start_timestamps)):]
    #     val_ids = list(set(start_timestamps['Case ID'])-(set(train_ids).union(set(test_ids))))
    #     # train_idx = [int(x) for x in train_ids]
    #     # test_idx = [int(x) for x in test_ids]
    #     # val_idx = [int(x) for x in val_ids]
    #     train_idx = df.index[df["Case ID"].isin(train_ids)].tolist()
    #     val_idx = df.index[df["Case ID"].isin(val_ids)].tolist()
    #     test_idx = df.index[df["Case ID"].isin(test_ids)].tolist()
    #     if data_format.lower() == PANDAS_SINGLE:
    #         return df
    #     else:
    #         w = df.drop(['treatment', 'outcome'], axis='columns')
    #         t = df['treatment']
    #         y = df['outcome']
    #     w,t,y = to_data_format(data_format, w, t, y)
    #     return w,t,y, train_idx, val_idx, test_idx
        

def load_bpi17(DATA_NAME, dataroot=None, ):
    if dataroot is None:
        dataroot = DATA_FOLDER

    df = pd.read_csv(os.path.join("..", dataroot, '%s_encoded.csv'%DATA_NAME), sep=';')
    df = df.dropna()
    return df#pd.read_csv(os.path.join("..", dataroot, 'train_encoded_bpic2017.csv'), sep=';')
    #return pd.read_pickle(os.path.join("..", dataroot, 'bpi17.pkl'))

def load_bpi12(DATA_NAME, dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    df = pd.read_csv(os.path.join("..", dataroot, '%s_encoded.csv'%DATA_NAME), sep=';')
    df = df.dropna()
    return df

def load_bpi16(dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    return pd.read_pickle(os.path.join("..", dataroot, 'bpi16.pkl'))


def load_bpi20(dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    return pd.read_csv(os.path.join("..", dataroot, 'DomesticDeclarations.csv'))

def load_bpi19(dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    return pd.read_pickle(os.path.join("..", dataroot, 'bpi19.pkl'))

def load_bpi22(dataroot=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    return pd.read_pickle(os.path.join(dataroot, 'data_17_multiple_offers_test_newRatio.pkl'))

# def load_bpi12(dataroot=None):
#     if dataroot is None:
#         dataroot = DATA_FOLDER
#     return pd.read_pickle(os.path.join(dataroot, 'data_12_multiple_offers_test_newRatio.pkl'))