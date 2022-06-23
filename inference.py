import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path

import scipy as sp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

INPUT_DIR = './data/'
OUTPUT_DIR = './output'

class CFG:
    num_workers=4
    path="./output/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-base"
    batch_size=32
    fc_dropout=0.2
    target_size=1
    max_len=133
    seed=42
    n_fold=4
    use_mixout=False
    trn_fold=[0, 1, 2, 3]


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER = get_logger()
seed_everything(seed=42)

oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
labels = oof_df['score'].values
preds = oof_df['pred'].values
score = get_score(labels, preds)
LOGGER.info(f'CV Score: {score:<.4f}')

test = pd.read_csv(INPUT_DIR+'test.csv')
submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')

cpc_texts = torch.load(CFG.path+"cpc_texts.pth")
test['context_text'] = test['context'].map(cpc_texts)
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs

class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x, mask):
        w = self.attention(x).float() #
        w[mask==0]=float('-inf')
        w = torch.softmax(w,1)
        x = torch.sum(w * x, dim=1)
        return x

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path = None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        #model config information
        if config_path is None:
            self.config = AutoConfig.from_pretrained(self.cfg.model, output_hidden_states = True)
        else:
            self.config = torch.load(config_path)
            
        if pretrained:
            self.model = AutoModel.from_pretrained(self.cfg.model, config = self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        
        self.pool = AttentionPool(self.config.hidden_size)
            
            
        if self.cfg.use_mixout:
            for sup_module in self.model.modules():
                for name, module in sup_module.named_children():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    if isinstance(module, nn.Linear):
                        target_state_dict = module.state_dict()
                        bias = True if module.bias is not None else False
                        new_module = MixLinear(
                            module.in_features, module.out_features, bias, 
                            target_state_dict["weight"], self.cfg.mixout_prob
                        )
                        new_module.load_state_dict(target_state_dict)
                        setattr(sup_module, name, new_module)    
        
            
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        """
        self.fc = ArcMarginProduct(in_features = self.config.hidden_size, 
                                   out_features = self.cfg.target_size,
                                   s = self.cfg.s, 
                                   m = self.cfg.m, 
                                   easy_margin = self.cfg.ls_eps, 
                                   ls_eps = self.cfg.ls_eps
                                  )
        """
        self._init_weights(self.fc)
        self._init_weights(self.pool)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, inputs):
        #print(inputs)
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        x = self.pool(last_hidden_states, inputs['attention_mask'])
        x = self.fc_dropout(x)
        x = self.fc(x).reshape(-1)
        return x

def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
predictions = []
for fold in CFG.trn_fold:
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    predictions.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()
predictions = np.mean(predictions, axis=0)
submission['score'] = predictions
submission[['id', 'score']].to_csv('submission.csv', index=False)