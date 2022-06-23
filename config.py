import torch

class CFG:
    wandb=True
    competition='PPPM'
    _wandb_kernel='chowk'
    debug=False
    apex=True
    print_freq=100
    num_workers=4
    model="xlm-roberta-large" #"microsoft/deberta-v3-base"#'cross-encoder/ms-marco-electra-base' #"AI-Growth-Lab/PatentSBERTa" #"microsoft/deberta-v3-large"
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=16
    fc_dropout=0.2
    target_size=1
    max_len=256
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    mixout_prob = 0.2
    use_mixout = True
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    train=True
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_name = f"{model}_model_mixout_{use_mixout}"
    
    #Arcface
    s = 30.0 
    m = 0.50
    ls_eps = 0.0
    easy_margin = False
    embedding_size = 256