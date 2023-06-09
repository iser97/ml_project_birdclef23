import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sklearn
from optuna_utils.config import CFG
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor
import os
import librosa


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()

def set_device(batch: list, device):
    for index, item in enumerate(batch):
        batch[index] = item.to(device)
    return batch

def measurement(y_true, y_pred, padding_factor=5):
    if not CFG.loss=='BCE':
        y_true = F.one_hot(torch.from_numpy(y_true), num_classes=CFG.num_classes).numpy()
    # y_true = y_true.numpy()
    num_classes = y_true.shape[1]
    pad_rows = np.array([[1]*num_classes]*padding_factor)
    y_true = np.concatenate([y_true, pad_rows])
    y_pred = np.concatenate([y_pred, pad_rows])
    score = sklearn.metrics.average_precision_score(y_true, y_pred, average='macro',)
    roc_aucs = sklearn.metrics.roc_auc_score(y_true, y_pred, average='macro')
    return score, roc_aucs

def cal_gpu(module):
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    for submodule in module.children():
        if hasattr(submodule, "_parameters"):
            parameters = submodule._parameters
            if "weight" in parameters:
                return parameters["weight"].device

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def extract_ast_features(df):
    ast_feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    paths = df.filepath.values
    for path in tqdm(paths):
        if os.path.exists(path.replace('.ogg', '.pt')):
            continue
        raw, sr = librosa.load(path, sr=CFG.sample_rate, mono=True)
        raw = librosa.resample(raw, orig_sr=CFG.sample_rate, target_sr=16000)  # ast can only process the audio with sr=16000
        inputs = ast_feature_extractor(raw, sampling_rate=16000, return_tesnors='pt')
        input_values = inputs['input_values']
        torch.save(input_values, path.replace('.ogg', '.pt'))

def extract_audio_feature(df):
    paths = df.filepath.values
    for path in tqdm(paths):
        if os.path.exists(path.replace('.ogg', '.npy')):
            continue
        raw, sr = librosa.load(path, sr=CFG.sample_rate, mono=True)
        np.save(path.replace('.ogg', '.npy'), raw)

def extract_mfcc_feature(df):
    paths = df.filepath.values
    for path in tqdm(paths):
        save_path = path.replace('.ogg', '_mfcc.npy')
        if os.path.exists(save_path):
            continue
        y, sr = librosa.core.load(path, sr=CFG.sample_rate, mono=True)
        spec = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=CFG.sample_rate, n_fft=512, hop_length=256, n_mels=128))
        mfcc = librosa.feature.mfcc(S=spec, n_mfcc=128)
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_dd = librosa.feature.delta(mfcc, order=2)
        mfcc_stack = np.stack([mfcc, mfcc_d, mfcc_dd])
        np.save(save_path, mfcc_stack)