import os
import json
import zipfile
from pathlib import Path
import pandas as pd
print("before torch")
import torch
print("loading.py")
from addict import Dict


from train_generator import get_args, main
from consts import REALCAUSE_DATASETS_FOLDER, N_SAMPLE_SEEDS

DATA_NAME = get_args().parse_args().data.lower()

print("+++++++++++++")
print(DATA_NAME)
print(DATA_NAME)
if DATA_NAME=="bpic17":
    DATA_NAME="bpic2017"
elif DATA_NAME=="bpic12":
    DATA_NAME="bpic2012"

import os
dirname = os.path.dirname(__file__)
print(dirname)

prepared_data = os.path.join(dirname, "./../prepared_data/%s/"%DATA_NAME)
checkpoint_path= os.path.join(dirname, "./../results_realcause_%s/"%DATA_NAME)
model_folder = os.path.join(dirname, "./../results_realcause_%s/"%DATA_NAME)
#model_folder = "./../results_realcause_%s/"%DATA_NAME #subdata_path / Path(subfolders[0])

#"./results_realcause_%s/"%DATA_NAME

def load_realcause_dataset(dataset, sample=0):
    valid_datasets = {'lalonde_cps', 'lalonde_psid', 'twins', 'bpic17', 'bpic12'}
    dataset = dataset.lower()
    if dataset not in valid_datasets:
        raise ValueError('Invalid dataset "{}" ... Valid datasets: {}'
                         .format(dataset, valid_datasets))
    if not isinstance(sample, int):
        raise ValueError('sample must be an integer')
    if 0 < sample >= N_SAMPLE_SEEDS:
        raise ValueError('sample must be between 0 and {}'
                         .format(N_SAMPLE_SEEDS - 1))

    dataset_file = Path(REALCAUSE_DATASETS_FOLDER) / '{}_sample{}.csv'.format(dataset, sample)
    return pd.read_csv(dataset_file)


def load_gen(saveroot='save', dataroot=None):
    args = get_args().parse_args([])
    print(args)
    args_path = os.path.join(saveroot, 'args.txt')
    print(args_path)
    args.__dict__.update(json.load(open(args_path, 'r')))
    print(args)

    # overwriting args
    args.train = False
    args.eval = False
    args.comet = False
    args.saveroot = saveroot
    args.comet = False
    if dataroot is not None:
        args.dataroot = dataroot

    # initializing model
    model = main(args, False, False)

    # loading model params
    print(model.savepath)
    state_dicts = torch.load(model.savepath)
    for state_dict, net in zip(state_dicts, model.networks):
        net.load_state_dict(state_dict)
    return model, args


def load_from_folder(dataset, checkpoint_path=checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    root_dir = checkpoint_path.parent
    checkpoint_dir = root_dir / checkpoint_path.stem

    if not checkpoint_dir.is_dir():
        with zipfile.ZipFile(checkpoint_path, "r") as zip_ref:
            zip_ref.extractall()

    dataset_roots = os.listdir(checkpoint_dir)
    print(dataset_roots)
    print(dataset)
    dataset_stem = dataset.split("_")[0]
    
    subdata_stem = dataset.split("_")[-1]

    print(f"dataset_stem: {dataset_stem}")
    print(f"dataset_roots: {dataset_roots}")



    with open(model_folder + "args.txt") as f:
        args = Dict(json.load(f))

    args.saveroot = checkpoint_path
    args.dataroot =  prepared_data
    args.comet = False

    # Now load model
    model, args = load_gen(saveroot=str(args.saveroot), dataroot=prepared_data)
    return model, args


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            return f