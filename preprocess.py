import re
import os
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from config import CFG

def get_cpc_texts():
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in os.listdir('./data/cpc-data/CPCSchemeXML202105'):
        result = re.findall(pattern, file_name) # B01
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(f'./data/cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt') as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results


def preprocess_df(train, test, OUTPUT_DIR):
    cpc_texts = get_cpc_texts()
    torch.save(cpc_texts, OUTPUT_DIR+"cpc_texts.pth")
    train['context_text'] = train['context'].map(cpc_texts)
    test['context_text'] = test['context'].map(cpc_texts)

    train['text'] = train['anchor'] + '[SEP]' + train['target'] + '[SEP]'  + train['context_text']
    test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
    train['score_map'] = train['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})

    Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['score_map'])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)

    return train, test