import tokenizers
import transformers
from config import CFG
from mixout import *

import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features, s = 3.0, m = 0.5, easy_margin=False, ls_eps=0.0):
#         super(ArcMarginProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.ls_eps = ls_eps
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
        
#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th  = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
        
#     def forward(self, input, labels):
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m

#         #cosine = cosine.float()
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        
#         one_hot = torch.zeros(cosine.size(), device = device)
#         one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
#         if self.ls_eps > 0:
#             one_hot = (1-self.ls_eps) * one_hot + self.ls_eps / self.out_features
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.s
        
#         return output

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
    def __init__(self, cfg, device,  config_path = None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.device = device
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
                    elif isinstance(module, nn.Linear):
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