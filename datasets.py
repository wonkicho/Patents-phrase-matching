from config import CFG
import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values
        self.labels = df['score'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = self.prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label

    def prepare_input(self, cfg, text):
        inputs = cfg.tokenizer(text, add_special_tokens = True, max_length = cfg.max_len, padding="max_length", return_offsets_mapping = False)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs


