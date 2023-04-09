from optuna_utils.config import CFG
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio_augmentations import *
import pandas as pd
from prefetch_generator import BackgroundGenerator
import librosa
import torchaudio
import os

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def audio_augmentation(audio_data : torch.Tensor, sr: int, 
                       num_augmented_sampels=4,
                       num_samples=16000,
                       polarity_p=0.8, 
                       min_snr=0.001, 
                       max_snr=0.005, 
                       noise_p=0.3, 
                       gain_p=0.2, 
                       delay_p=0.5, 
                       shift_p=0.4, 
                       reverb_p=0.3):
    if num_samples>max(audio_data.shape):
        num_samples = max(audio_data.shape)
    
    transforms = [
                RandomResizedCrop(n_samples=num_samples),
                RandomApply([PolarityInversion()], p=polarity_p),
                RandomApply([Noise(min_snr=min_snr, max_snr=max_snr)], p=noise_p),
                RandomApply([Gain()], p=gain_p),
                HighLowPass(sample_rate=sr), # this augmentation will always be applied in this aumgentation chain!
                RandomApply([Delay(sample_rate=sr)], p=delay_p),
                RandomApply([PitchShift(
                    n_samples=num_samples,
                    sample_rate=sr
                )], p=shift_p),
                RandomApply([Reverb(sample_rate=sr)], p=reverb_p)]   
    transforms = ComposeMany(transforms=transforms, num_augmented_samples=num_augmented_sampels)
    audio_aug = transforms(audio_data)
    return audio_aug

def filter_data(df, thr=5):
    # Count the number of samples for each class
    counts = df.primary_label.value_counts()

    # Condition that selects classes with less than `thr` samples
    cond = df.primary_label.isin(counts[counts<thr].index.tolist())

    # Add a new column to select samples for cross validation
    df['cv'] = True

    # Set cv = False for those class where there is samples less than thr
    df.loc[cond, 'cv'] = False

    # Return the filtered dataframe
    return df
    
def upsample_data(df, thr=20):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df

def audio_fixlength(raw, input_length: int):
    length = len(raw)
    factor = input_length // length + 1
    raw = [raw for _ in range(factor)]
    if type(raw[0])==np.ndarray:
        raw = np.stack(raw, axis=0)
        raw = raw.flatten()
    elif type(raw[0])==torch.Tensor:
        raw = torch.stack(raw, dim=0)
        raw = raw.flatten()
    return raw

class AudioDataset(Dataset):
    def __init__(self, df, fold=4, mode='train', transform=None):
        self.mode = mode
        if CFG.cv_filter:
            train_df = df.query("fold!={} | ~cv".format(fold)).reset_index(drop=True)
            valid_df = df.query("fold=={} & cv".format(fold)).reset_index(drop=True)
        else:
            train_df = df.query("fold!={}".format(fold)).reset_index(drop=True)
            valid_df = df.query("fold=={}".format(fold)).reset_index(drop=True)
        ori_train_df_path = train_df.filepath.values
        ori_valid_df_path = valid_df.filepath.values
        self.input_length = CFG.sample_rate * CFG.time_length  # choose 4 seconds samples in training
        # Upsample train data
        train_df = upsample_data(train_df, thr=CFG.upsample_thr)
        
        # Get file paths and labels
        self.train_paths = train_df.filepath.values
        self.train_labels = train_df.target.values
        self.valid_paths = valid_df.filepath.values
        self.valid_labels = valid_df.target.values
        
        if CFG.debug:
            self.train_paths = self.train_paths[:CFG.batch_size]
            self.train_labels = self.train_labels[:CFG.batch_size]
            self.valid_paths = self.valid_paths[:CFG.batch_size]
            self.valid_labels = self.valid_labels[:CFG.batch_size]
            
        self.transform = transform
        self.audio_files = {}
        
        if self.mode=='train':
            if not os.path.exists('train_audio_file_{}khz.npy'.format(CFG.target_rate)):
                for path in ori_train_df_path:
                    audio = librosa.load(path, sr=CFG.sample_rate, mono=True)[0]
                    audio = librosa.resample(y=audio, orig_sr=CFG.sample_rate, target_sr=CFG.target_rate)
                    self.audio_files[path] = audio
                np.save('train_audio_file_{}khz.npy'.format(CFG.target_rate), np.array(self.audio_files, dtype=object), allow_pickle=True)
            else:
                self.audio_files = np.load('train_audio_file_{}khz.npy'.format(CFG.target_rate), allow_pickle=True).item()
                
            # for path in self.train_paths:
            #     audio = librosa.load(path, sr=CFG.sample_rate, mono=True)[0]
            #     audio = librosa.resample(y=audio, orig_sr=CFG.sample_rate, target_sr=CFG.target_rate)
            #     self.audio_files[path] = audio

        elif self.mode=='eval':

            if not os.path.exists('eval_audio_file_{}khz.npy'.format(CFG.target_rate)):
                for path in ori_valid_df_path:
                    audio = librosa.load(path, sr=CFG.sample_rate, mono=True)[0]
                    audio = librosa.resample(y=audio, orig_sr=CFG.sample_rate, target_sr=CFG.target_rate)
                    self.audio_files[path] = audio
                np.save('train_audio_file_{}khz.npy'.format(CFG.target_rate), np.array(self.audio_files, dtype=object), allow_pickle=True)
            else:
                self.audio_files = np.load('eval_audio_file_{}khz.npy'.format(CFG.target_rate), allow_pickle=True).item()

            # for path in self.valid_paths:
            #     audio = librosa.load(path, sr=CFG.sample_rate, mono=True)[0]
            #     audio = librosa.resample(y=audio, orig_sr=CFG.sample_rate, target_sr=CFG.target_rate)
            #     self.audio_files[path] = audio
        
        self.input_length = CFG.target_rate * CFG.time_length
        CFG.sample_rate = CFG.target_rate
    
    def __len__(self):
        return max(len(self.train_paths), len(self.valid_paths))

    def __getitem__(self, idx):
        if self.mode=='train':
            audio_path = self.train_paths[idx%len(self.train_paths)]
            label = self.train_labels[idx%len(self.train_paths)]
        else:
            audio_path = self.valid_paths[idx%len(self.valid_paths)]
            label = self.valid_labels[idx%len(self.valid_paths)]
        
        # sig, sr = librosa.load(audio_path, sr=CFG.sample_rate, mono=True)
        # sig = np.load(audio_path.replace('.ogg', '.npy'))
        sig = self.audio_files[audio_path]
        if len(sig) < self.input_length:
            sig = audio_fixlength(sig, self.input_length)
            
        if self.mode=='test':  
            hop = (len(sig) - self.input_length) // CFG.batch_size
            x = [torch.Tensor(np.float32(sig[i*hop:i*hop+self.input_length])) for i in range(CFG.batch_size)] 
            x = torch.stack(x)
            padding_mask = torch.zeros(CFG.batch_size, x.shape[1]).bool()
            return x, padding_mask, label
    
        elif self.mode=='train' or self.mode=='eval':
            sig_t = torch.tensor(sig)
            sig_t = sig_t.unsqueeze(0)
            sig_t = audio_augmentation(sig_t, CFG.sample_rate, num_augmented_sampels=1, num_samples=self.input_length)
            sig_t = sig_t.squeeze()
            padding_mask = torch.zeros(1, sig_t.shape[0]).bool().squeeze(0)
            if self.transform:
                sig_t = self.transform(sig_t)
            return sig_t, padding_mask, label


class ASTDataset(AudioDataset):
    def __init__(self, df, fold=4, mode='train', transform=None):
        super().__init__(df, fold, mode, transform)
        self.train_paths = [path.replace('.ogg', '.pt') for path in self.train_paths]
        self.valid_paths = [path.replace('.ogg', '.pt') for path in self.valid_paths]
    
    def __getitem__(self, idx):
        if self.mode=='train':
            audio_path = self.train_paths[idx%len(self.train_paths)]
            label = self.train_labels[idx%len(self.train_paths)]
        else:
            audio_path = self.valid_paths[idx%len(self.valid_paths)]
            label = self.valid_labels[idx%len(self.valid_paths)]
        feature = torch.load(audio_path)[0]
        return feature, label


class MusicnnDataset(AudioDataset):
    def __init__(self, df, fold=4, mode='train', transform=None):
        super().__init__(df, fold, mode, transform)
        self.input_length = CFG.sample_rate * CFG.time_length / 256
        self.train_paths = [path.replace('.ogg', '_mfcc.npy') for path in self.train_paths]
        self.valid_paths = [path.replace('.ogg', '_mfcc.npy') for path in self.valid_paths]

    def audio2mfcc(self, audio):
        spec = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(y=audio, sr=CFG.sample_rate, n_fft=512, hop_length=256, n_mels=128))
        mfcc = librosa.feature.mfcc(S=spec, n_mfcc=128)
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_dd = librosa.feature.delta(mfcc, order=2)
        mfcc_stack = np.stack([mfcc, mfcc_d, mfcc_dd])
        return mfcc_stack
    
    def __getitem__(self, idx):
        if self.mode=='train':
            audio_path = self.train_paths[idx%len(self.train_paths)]
            label = self.train_labels[idx%len(self.train_paths)]
        else:
            audio_path = self.valid_paths[idx%len(self.valid_paths)]
            label = self.valid_labels[idx%len(self.valid_paths)]
        
        # mfcc_stack = np.load(audio_path)
        mfcc_stack = self.audio2mfcc(self.audio_files[audio_path.replace('_mfcc.npy', '.ogg')])
        if mfcc_stack.shape[2]<self.input_length:
            mfcc_stack = self.mfcc_expand(mfcc_stack, self.input_length)
            
        if self.mode=='train' or self.mode=='eval':
            random_idx = int(np.floor(np.random.random(1) * (mfcc_stack.shape[2]-self.input_length)))
            data = np.array(mfcc_stack[:, :, random_idx:random_idx+self.input_length], dtype=np.float32)
        else:
            channel, mfcc_filters, length = mfcc_stack.shape
            hop = (length - self.input_length) // CFG.batch_size
            data = torch.zeros(CFG.batch_size, channel, mfcc_filters, self.input_length)
            for i in range(CFG.batch_size):
                data[i] = torch.Tensor(mfcc_stack[:, :, i*hop:i*hop+self.input_length])
        return data, label
    
    @staticmethod
    def mfcc_expand(mfcc, input_length):
        length = mfcc.shape[2]
        factor = input_length // length + 1
        start = mfcc
        for i in range(factor):
            start = np.concatenate([start, mfcc], axis=2)
        start = start[:, :, :input_length]
        return start        