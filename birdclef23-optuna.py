import optuna
from optuna.trial import TrialState
import os
import pandas as pd
pd.options.mode.chained_assignment = None # avoids assignment warning
import numpy as np
import random
from glob import glob
from tqdm import tqdm
tqdm.pandas()  # enable progress bars in pandas operations
import gc

import librosa
import sklearn
import json
import argparse

# Import for visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa.display as lid
import IPython.display as ipd

# from kaggle_datasets import KaggleDatasets

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from torchvision import transforms
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

from optuna_utils.config import CFG
from optuna_utils.dataset import AudioDataset, ASTDataset, filter_data, DataLoaderX, MusicnnDataset
from optuna_utils.models import BirdModel, ASTagModel, Musicnn
from transformers import set_seed
from transformers import AutoConfig

set_seed(CFG.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
if CFG.debug:
    device = torch.device('cpu')
    CFG.use_apex = False

def main(args):
    if args.model_name=='beats':
        dataset_train = AudioDataset(df, fold=args.fold, mode='train')
        # loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0 if CFG.debug else 10)
        loader_train = DataLoaderX(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0 if CFG.debug else 10)
        dataset_eval = AudioDataset(df, fold=args.fold, mode='eval')
        # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=0 if CFG.debug else 10)
        loader_eval = DataLoaderX(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=0 if CFG.debug else 10)
        model = BirdModel(args)
    elif args.model_name=='ast':
        dataset_train = ASTDataset(df, fold=args.fold, mode='train')
        loader_train = DataLoader(dataset_train, batch_size=CFG.batch_size, shuffle=True, num_workers=0 if CFG.debug else 10)
        dataset_eval = ASTDataset(df, fold=args.fold, mode='eval')
        loader_eval = DataLoader(dataset_eval, batch_size=CFG.batch_size, shuffle=False, num_workers=0 if CFG.debug else 10)

        config = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")  
        model = ASTagModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            train_config=args
        )
    elif args.mode_name=='musicnn':
        dataset_train = MusicnnDataset(df, fold=args.fold, mode='train')
        loader_train = DataLoaderX(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0 if CFG.debug else 10)
        dataset_eval = MusicnnDataset(df, fold=args.fold, mode='eval')
        loader_eval = DataLoaderX(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=0 if CFG.debug else 10)
        model = Musicnn(args)
    else:
        raise ValueError('The model type - {} has not been implemented'.format(args.model_name))
    
    model = model.to(device)
    total_samples = dataset_train.__len__()
    num_warmup_steps = (total_samples // args.batch_size) * 2
    num_total_steps = (total_samples // args.batch_size) * args.max_epoch
    lr_scheduler = get_cosine_schedule_with_warmup(model.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps)

    best_metric = 0
    for epoch in tqdm(range(args.max_epoch)):
        model.train_step(loader_train, lr_scheduler)
        best_metric = model.eval_step(args, loader_eval, best_metric, epoch, model_name='beats.pth')
    return best_metric


def objective(trial):
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_int('batch size', 32, 64)
    args.fold = trial.suggest_int('split fold', 0, 4)
    CFG.batch_size = args.batch_size
    CFG.lr = args.lr
    
    args.save_dir = os.path.join(experiment_dir, "trial_{}".format(trial.number))
    os.makedirs(args.save_dir, exist_ok=True)
    
    for key, value in trial.params.items():
        print("  {}: {} \n".format(key, value))
    best_metric = main(args)
    
    with open(os.path.join(args.save_dir, "best_metric.txt"), mode='w', encoding='utf-8') as w:
        for key, value in trial.params.items():
            w.writelines("    {}: {} \n".format(key, value))
        w.writelines(str(best_metric))
    return best_metric

def do_trial(args):
    study = optuna.create_study(directions=['maximize'])
    study.optimize(objective, n_trials=args.n_trials)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
if __name__ == '__main__':
        
    if CFG.device=="TPU":
        from kaggle_datasets import KaggleDatasets
        GCS_PATH = KaggleDatasets().get_gcs_path(CFG.base_path.split('/')[-1])
    else:
        GCS_PATH = CFG.base_path
    
    df = pd.read_csv(f'{CFG.base_path}/train_metadata.csv')
    df['filepath'] = GCS_PATH + '/train_audio/' + df.filename
    df['target'] = df.primary_label.map(CFG.name2label)

    f_df = filter_data(df, thr=5)
    f_df.cv.value_counts().plot.bar(legend=True)
    

    # Initialize the StratifiedKFold object with 5 splits and shuffle the data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

    # Reset the index of the dataframe
    df = df.reset_index(drop=True)

    # Create a new column in the dataframe to store the fold number for each row
    df["fold"] = -1

    # Iterate over the folds and assign the corresponding fold number to each row in the dataframe
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        df.loc[val_idx, 'fold'] = fold


    parser = argparse.ArgumentParser(description='PyTorch Kaggle Bird Implementation')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--max_epoch', type=int, default=25, metavar='N',
                        help='how many epochs')
    parser.add_argument('--experiment_name', type=str, default='beats',
                        help='experiment name')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='number of trials')
    parser.add_argument('--best_acc', type=float, default=0,
                        help='number of trials')
    parser.add_argument('--model_name', type=str, default='beats', choices=['beats', 'ast', 'musicnn'])
    parser.add_argument('--eval_step', type=int, default=1)
    args = parser.parse_args()
    
    experiment_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=False)
    do_trial(args)
    print(args)
    