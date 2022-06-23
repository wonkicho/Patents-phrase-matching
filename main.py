from config import CFG
from utils import *
from preprocess import *
from train_fn import *

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import pandas as pd
import wandb
import argparse
import os

from datasets import TrainDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type = str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main():
    wandb.init(project = CFG.competition, entity=CFG._wandb_kernel, name = CFG.wandb_name, reinit=True)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"


    INPUT_DIR = "./data/"
    OUTPUT_DIR = './output/'
    LOGGER = get_logger(OUTPUT_DIR + "train")
    seed_everything(seed=42)

    
    

    train = pd.read_csv(INPUT_DIR + 'train.csv')
    test = pd.read_csv(INPUT_DIR+'test.csv')
    submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')

    train, test = preprocess_df(train, test, OUTPUT_DIR = OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer = tokenizer

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold, LOGGER, device, OUTPUT_DIR, CFG)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, LOGGER)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df, LOGGER)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')

if __name__ == "__main__":
    #opt = parse_opt(True)
    #os.environ["CUDA_VISIBLE_DEVICES"]="1,3"
    main()