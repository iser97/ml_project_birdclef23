from optuna_utils.config import CFG
from optuna_utils.utils import *
import torch.nn as nn
import torch
import numpy as np
from BEATs.BEATs import BEATs, BEATsConfig
from optuna_utils.modules import *
import torch.nn.functional as F
from torch.autograd import Variable
import sklearn
import random
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import ASTPreTrainedModel, ASTModel, AutoConfig, AutoFeatureExtractor
import os
# from efficientnet_pytorch import EfficientNet
from optuna_utils.efficientnet_model import EfficientNet

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
        self._build_model()
        
        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
        if CFG.loss=='CCE':
            try:
                self.loss_fct = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
            except:
                self.loss_fct = nn.CrossEntropyLoss()
        elif CFG.loss=='BCE':
            self.loss_fct = nn.BCELoss()
        elif CFG.loss=='focal':
            self.loss_fct = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.from_numpy(CFG.class_weights).type(torch.float32),
            gamma=2,
            reduction='mean',
            force_reload=False
            )

        if CFG.use_apex:
            self.scaler = GradScaler()
        else:
            self.scaler = None

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

        return nn.Sigmoid()(x) if CFG.loss=='BCE' else x
    
    def train_step(self, loader, lr_scheduler):
        device = cal_gpu(self)
        self.train()
        for batch in loader:
            self.optimizer.zero_grad()
            batch = set_device(batch, device)
            audio, mask, label = batch
            if random.random()<CFG.mixup_prob:
                inputs, targets_a, targets_b, lam = mixup_data(audio, label, CFG.mixup_alpha)
                inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
                outputs = self.forward(inputs, mask)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(self.loss_fct, outputs)
            else:
                prob = self.forward(audio, mask)
                loss = self.loss_fct(prob, label)
            if CFG.use_apex:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            lr_scheduler.step()
    
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
        if cur_epoch%args.eval_step==0:
            print("cur loss: {:.4} --- acc: {:.4} --- auc: {:.4}".format(cur_loss, acc, auc))
        if acc>best_metric:
            best_metric = acc
            torch.save(self.state_dict(), os.path.join(args.save_dir, model_name))
        return best_metric


class ASTagModel(ASTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.train_config = kwargs['train_config']
        self.audio_spectrogram_transformer = ASTModel(config)
        for pname, p in self.named_parameters():
            if pname.find('layer.') >= 0:
                layer = int(pname.split('.')[3])
                if layer<=CFG.ast_fix_layer:
                    p.requires_grad = False
            else:
                p.requires_grad = False

        self.linear = DenseLayer(config, CFG.num_classes)
        self.n_class = CFG.num_classes
    
        self.optimizer = torch.optim.Adam(self.parameters(), self.train_config.lr)
        if CFG.loss=='CCE':
            try:
                self.loss_fct = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
            except:
                self.loss_fct = nn.CrossEntropyLoss()
        elif CFG.loss=='BCE':
            self.loss_fct = nn.BCELoss()
        elif CFG.loss=='focal':
            self.loss_fct = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.from_numpy(CFG.class_weights).type(torch.float32),
            gamma=2,
            reduction='mean',
            force_reload=False
            )
        if CFG.use_apex:
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def forward(self, input_values):
        outputs = self.audio_spectrogram_transformer(input_values)
        hidden_states = outputs.last_hidden_state
        pool_output = torch.mean(hidden_states, dim=1)
        # pool_output = outputs.pooler_output
        logits = self.linear(pool_output)
        return nn.Sigmoid()(logits) if CFG.loss=='BCE' else logits

    def train_step(self, loader, lr_scheduler):
        device = self.audio_spectrogram_transformer.device
        self.train()
        for batch in loader:
            self.optimizer.zero_grad()
            batch = set_device(batch, device)
            audio, label = batch
            if random.random()<CFG.mixup_prob:
                inputs, targets_a, targets_b, lam = mixup_data(audio, label, CFG.mixup_alpha)
                inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
                outputs = self.forward(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(self.loss_fct, outputs)
            else:
                prob = self.forward(audio)
                loss = self.loss_fct(prob, label)
            if CFG.use_apex:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            lr_scheduler.step()
    
    def eval_step(self, args, loader, best_metric, cur_epoch, model_name='pytorch_model.pth'):
        self.eval()
        device = self.audio_spectrogram_transformer.device
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
        if cur_epoch%args.eval_step==0:
            print("cur loss: {:.4} --- acc: {:.4} --- auc: {:.4}".format(cur_loss, acc, auc))
        if acc>best_metric:
            best_metric = acc
            torch.save(self.state_dict(), os.path.join(args.save_dir, model_name))
        return best_metric

class Musicnn(nn.Module):
    '''
    Pons et al. 2017
    End-to-end learning for music audio tagging at scale.
    This is the updated implementation of the original paper. Referred to the Musicnn code.
    https://github.com/jordipons/musicnn
    '''
    def __init__(self,
                 args,
                dataset='mtat'):
        super(Musicnn, self).__init__()
        # Spectrogram
        self.spec_bn = nn.BatchNorm2d(3)

        # Pons front-end
        m1 = Conv_V(3, 204, (int(0.7*96), 7))
        m2 = Conv_V(3, 204, (int(0.4*96), 7))
        m3 = Conv_H(3, 51, 129)
        m4 = Conv_H(3, 51, 65)
        m5 = Conv_H(3, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        backend_channel= 512 if dataset=='msd' else 64
        self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

        # Dense
        dense_channel = 500 if dataset=='msd' else 200
        self.dense1 = nn.Linear((561+(backend_channel*3))*2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, CFG.num_classes)
        
        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
        if CFG.loss=='CCE':
            try:
                self.loss_fct = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
            except:
                self.loss_fct = nn.CrossEntropyLoss()
        elif CFG.loss=='BCE':
            self.loss_fct = nn.BCELoss()
        elif CFG.loss=='focal':
            self.loss_fct = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.from_numpy(CFG.class_weights).type(torch.float32),
            gamma=2,
            reduction='mean',
            force_reload=False
            )
        if CFG.use_apex:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
    def forward(self, x):
        # Spectrogram
        # x = self.spec(x)
        # x = self.to_db(x)
        # x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out) if CFG.loss=='BCE' else out

        return out

    def train_step(self, loader, lr_scheduler):
        device = cal_gpu(self)
        self.train()
        for batch in loader:
            self.optimizer.zero_grad()
            batch = set_device(batch, device)
            audio, label = batch
            if random.random()<CFG.mixup_prob:
                inputs, targets_a, targets_b, lam = mixup_data(audio, label, CFG.mixup_alpha)
                inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
                outputs = self.forward(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(self.loss_fct, outputs)
            else:
                prob = self.forward(audio)
                loss = self.loss_fct(prob, label)
            if CFG.use_apex:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            lr_scheduler.step()
    
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
        if cur_epoch%args.eval_step==0:
            print("cur loss: {:.4} --- acc: {:.4} --- auc: {:.4}".format(cur_loss, acc, auc))
        if auc>best_metric:
            best_metric = auc
            torch.save(self.state_dict(), os.path.join(args.save_dir, model_name))
        return best_metric

class Efficient(Musicnn):
    def __init__(self, args):
        super(Efficient, self).__init__(args)
        self.args = args
        
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        feature = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features=feature, out_features=CFG.num_classes, bias=True)
        
        self.optimizer = torch.optim.Adam(self.parameters(), args.lr)
    
    def forward(self, audio):
        res = self.model(audio)
        res = nn.Sigmoid()(res) if CFG.loss=='BCE' else res
        return res
    
    
