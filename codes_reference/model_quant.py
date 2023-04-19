import torch
from torch.utils.data import DataLoader, Dataset
import os

from optuna_utils.dataset_pytorch import ASTDataset
from optuna_utils.config import CFG
from optuna_utils.dataset import filter_data, upsample_data
from sklearn.model_selection import StratifiedKFold
from transformers import ASTConfig
import pandas as pd
from transformers import ASTFeatureExtractor
from transformers import ASTPreTrainedModel, ASTModel, AutoConfig, ASTConfig
import torch.nn as nn
from tqdm import tqdm

class DenseLayer(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, output_dim)
    
    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state
    
class ASTagModel(ASTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # self.quant = torch.ao.quantization.QuantStub()
        self.quant = torch.quantization.QuantStub()
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
    
    def forward(self, input_values):
        input_values = self.quant(input_values)
        outputs = self.audio_spectrogram_transformer(input_values)
        hidden_states = outputs.last_hidden_state
        pool_output = torch.mean(hidden_states, dim=1)
        # pool_output = outputs.pooler_output
        logits = self.linear(pool_output)
        return nn.Sigmoid()(logits) if CFG.loss=='BCE' else logits
    
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') 

GCS_PATH = CFG.base_path

df = pd.read_csv(f'{CFG.base_path}/train_metadata.csv')
df['filepath'] = GCS_PATH + '/train_audio/' + df.filename
df['target'] = df.primary_label.map(CFG.name2label)

f_df = filter_data(df, thr=5)
f_df.cv.value_counts().plot.bar(legend=True)
up_df = upsample_data(df, thr=50)

CFG.class_weights = up_df.primary_label.value_counts()[:].to_numpy()

# Initialize the StratifiedKFold object with 5 splits and shuffle the data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

# Reset the index of the dataframe
df = df.reset_index(drop=True)

# Create a new column in the dataframe to store the fold number for each row
df["fold"] = -1

# Iterate over the folds and assign the corresponding fold number to each row in the dataframe
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
    df.loc[val_idx, 'fold'] = fold

dataset_eval = ASTDataset(df, fold=4, mode='eval')
loader_eval = DataLoader(dataset_eval, batch_size=10, shuffle=False, num_workers=0 if CFG.debug else 10)

example_inputs = [batch[0] for batch in loader_eval]


def evaluate(model):
    for batch in tqdm(loader_eval):
        audio, label = batch
        audio = audio.to(device)
        prob = model(audio)
        break
    

# state_dict = torch.load('experiments/ast_9layer/trial_1/ast.pth', map_location=device)
config = ASTConfig() 
model = ASTagModel(config=config)
model = model.to('cpu')
# model.load_state_dict(state_dict)
model.float()
model.eval()

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# model = torch.quantization.fuse_modules(model, [['conv', 'relu']])
model_prepared = torch.quantization.prepare(model, inplace=True, )

evaluate(model_prepared)

model_int8 = torch.quantization.convert(model_prepared)
# torch.save(model_int8, './ast_int8.pth')
torch.save(model_int8.state_dict(), './ast_int8_state.pth')

for batch in loader_eval:
    audio, label = batch
    model_int8(audio)

# ckpt = torch.load('./ast_int8_state.pth', map_location=device)

# config = ASTConfig() 
# model = ASTagModel(config)
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# model_fp32_prepared = torch.quantization.prepare(model)
# model_int8 = torch.quantization.convert(model_fp32_prepared)

# model_int8.load_state_dict(ckpt)
# model = model_int8
# model.eval()

# evaluate(model)


