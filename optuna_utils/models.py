from optuna_utils.config import CFG
import torch.nn as nn
import torch
import numpy as np
from BEATs.BEATs import BEATs, BEATsConfig
import torch.nn.functional as F
import sklearn
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import ASTPreTrainedModel, ASTModel, AutoConfig, AutoFeatureExtractor
import os

def set_device(batch: list, device):
    for index, item in enumerate(batch):
        batch[index] = item.to(device)
    return batch

def measurement(y_true, y_pred, padding_factor=5):
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

class BirdModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_target_classes = CFG.num_classes
        self.checkpoint = torch.load(CFG.unilm_model_path)
        self.cfg = BEATsConfig(
            {
                **self.checkpoint["cfg"],
                "predictor_class": self.num_target_classes,
                "finetuned_model": False,
            }
        )

        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
        if CFG.loss=='CCE':
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
        elif CFG.loss=='BCE':
            self.loss_fct = nn.BCELoss()

        if CFG.use_apex:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
        self._build_model()
        
    def _build_model(self):
        # 1. Load the pre-trained network
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])

        # 2. Classifier
        self.fc = nn.Linear(self.cfg.encoder_embed_dim, self.cfg.predictor_class)

    def forward(self, x, padding_mask=None):
        """Forward pass. Return x"""

        # Get the representation
        if padding_mask != None:
            x, _ = self.beats.extract_features(x, padding_mask, sample_frequency=CFG.sample_rate)
        else:
            x, _ = self.beats.extract_features(x, sample_frequency=CFG.sample_rate)

        # Get the logits
        x = self.fc(x)

        # Mean pool the second layer
        x = x.mean(dim=1)

        return x  
    
    def train_step(self, loader, lr_scheduler):
        device = cal_gpu(self)
        self.train()
        for batch in loader:
            batch = set_device(batch, device)
            audio, mask, label = batch
            prob = self.forward(audio, mask)
            loss = self.loss_fct(prob, label)
            if CFG.use_apex:
                self.scaler.scale(loss).backward()
                lr_scheduler.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
    
    def eval_step(self, args, loader, best_metric, cur_epoch, model_name='pytorch_model.pth'):
        self.eval()
        device = cal_gpu(self)
        label_stack = []
        pred_stack = torch.randn(size=(1, CFG.num_classes)).to(device)
        losses = []
        for batch in loader:
            batch = set_device(batch, device)
            audio, mask, label = batch
            with torch.no_grad():
                prob = self.forward(audio, mask)
                loss = self.loss_fct(prob, label)
                losses.append(loss.item())
            label_stack += label.cpu().numpy().tolist()
            pred_stack = torch.cat([pred_stack, prob], dim=0)
        cur_loss = np.array(losses).mean(0)
        pred_stack = pred_stack[1:]
        pred_stack = pred_stack.detach().cpu().numpy()
        label_stack = np.array(label_stack)
        acc, auc = measurement(label_stack, pred_stack)
        if cur_epoch%10==0:
            print("cur loss: {:.4} --- acc: {:.4} --- auc: {:.4}".format(cur_loss, acc, auc))
        if auc>best_metric:
            best_metric = auc
            torch.save(self.state_dict(), os.path.join(args.save_dir, model_name))
        return best_metric

class DenseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, CFG.num_classes)
    
    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state

class ASTagModel(ASTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.train_config = kwargs['train_config']
        self.audio_spectrogram_transformer = ASTModel(config)
        self.linear = DenseLayer(config)
        self.n_class = CFG.num_classes
    
        self.optimizer = torch.optim.Adam(self.parameters(), self.train_config.lr)
        if CFG.loss=='CCE':
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
        elif CFG.loss=='BCE':
            self.loss_fct = nn.BCELoss()

        if CFG.use_apex:
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def forward(self, input_values):
        outputs = self.audio_spectrogram_transformer(input_values)
        hidden_states = outputs.last_hidden_state
        # pool_output = torch.mean(hidden_states, dim=1)
        pool_output = outputs.pooler_output
        logits = self.linear(pool_output)
        return nn.Sigmoid()(logits)

    def train_step(self, loader, lr_scheduler):
        device = cal_gpu(self)
        self.train()
        for batch in loader:
            batch = set_device(batch, device)
            audio, label = batch
            prob = self.forward(audio)
            loss = self.loss_fct(prob, label)
            if CFG.use_apex:
                self.scaler.scale(loss).backward()
                lr_scheduler.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
    
    def eval_step(self, args, loader, best_metric, cur_epoch, model_name='pytorch_model.pth'):
        self.eval()
        device = cal_gpu(self)
        label_stack = []
        pred_stack = torch.randn(size=(1, CFG.num_classes)).to(device)
        losses = []
        for batch in loader:
            batch = set_device(batch, device)
            audio, label = batch
            with torch.no_grad():
                prob = self.forward(audio)
                loss = self.loss_fct(prob, label)
                losses.append(loss.item())
            label_stack += label.cpu().numpy().tolist()
            pred_stack = torch.cat([pred_stack, prob], dim=0)
        cur_loss = np.array(losses).mean(0)
        pred_stack = pred_stack[1:]
        pred_stack = pred_stack.detach().cpu().numpy()
        label_stack = np.array(label_stack)
        acc, auc = measurement(label_stack, pred_stack)
        if cur_epoch%10==0:
            print("cur loss: {:.4} --- acc: {:.4} --- auc: {:.4}".format(cur_loss, acc, auc))
        if auc>best_metric:
            best_metric = auc
            torch.save(self.state_dict(), os.path.join(args.save_dir, model_name))
        return best_metric

