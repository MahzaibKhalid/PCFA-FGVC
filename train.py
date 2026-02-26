import os
import shutil

import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from loguru import logger
import sys  # Add this line

from model.PACF import PACF_model
from utils.dataset import Aircraft_Dataset, CUB_Dataset, Car_Dataset, Dogs_Dataset, PACF_Collate
import torch.nn as nn
from torch.amp import GradScaler, autocast
import warnings
import config as cfg
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR,CosineAnnealingWarmRestarts


# import httpx

# # Set a custom timeout (e.g., 120 seconds)
# timeout = httpx.Timeout(120.0)

# # Create a custom HTTP client
# client = httpx.Client(timeout=timeout)

# # Set the Hugging Face mirror (example: from Europe)
# os.environ['HF_HOME'] = 'https://huggingface.co'

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

@logger.catch 
def main():
    device = torch.device(cfg.DEVICE)
    if cfg.DATASET_NAME=='Aircraft':
        train_data = Aircraft_Dataset(pattern="train", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
        test_data = Aircraft_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    elif cfg.DATASET_NAME=='CUB':
        train_data = CUB_Dataset(pattern="train", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
        test_data = CUB_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    elif cfg.DATASET_NAME=='Car':
        train_data = Car_Dataset(pattern="train", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
        test_data = Car_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    elif cfg.DATASET_NAME=='Dogs':
        train_data = Dogs_Dataset(pattern="train", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
        test_data = Dogs_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    else:
        warnings.warn("Undefined dataset", ImportWarning)
        return

    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=cfg.BS,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=False,
                                              collate_fn=PACF_Collate
                                              )
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=cfg.BS,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=False,
                                            collate_fn=PACF_Collate
                                            )

    part_name = cfg.PART_NAME
    model = PACF_model(stage=cfg.STAGE, embed_dim=cfg.EMBED_DIM, backbone=cfg.BACKBONE, dataset_name=cfg.DATASET_NAME, part_name=part_name, device=device, use_precomputed_masks=cfg.USE_PRECOMPUTED_MASKS, part_mask_path=cfg.PART_MASK_PATH,)
    
    # Add parameter counting here
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")

    part_name.insert(0, "Coarse")
    part_name.append("Sum")
    max_length = max(len(s) for s in part_name)
    part_name = [s.ljust(max_length) for s in part_name]
  
    for n, param in model.named_parameters():
        param.requires_grad = True
    if hasattr(model, 'clip_model'):
        for n, param in model.clip_model.named_parameters():
            param.requires_grad = False

    param_groups = [
        {'params': model.backbone.parameters()},
        {'params': model.cls_token}, 
        {'params': model.MSCFS.parameters()},
        {'params': model.norm.parameters()}, 
        {'params': model.transformer.parameters()},
    ]
    if hasattr(model, 'clip_loss'):
        param_groups.append({'params': model.clip_loss.parameters()})
    if hasattr(model, 'classifier'):
        param_groups.append({'params': model.classifier.parameters()})
    names = [n for n,_ in model.named_parameters() if _.requires_grad]
    print(f"✅ DEBUG[train]: optimizer includes clip_loss params = {any('clip_loss' in n for n in names)}; classifier params = {any('classifier' in n for n in names)}")
    optimizer = torch.optim.AdamW(param_groups,
                        lr=cfg.LR,
                        weight_decay=cfg.WD, 
                        eps=1e-8,
                        betas = (0.9, 0.95),
                        amsgrad=False)

    def lr_lambda_warmup(epoch):
        if epoch < cfg.WARMUP_NUM:
            return float(epoch + 1) / cfg.WARMUP_NUM
        else:
            return 1.0  
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCH_NUM - cfg.WARMUP_NUM, eta_min=1e-8)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.WARMUP_NUM]
    )
    scaler = GradScaler()
        
    continue_train = os.path.exists(cfg.LOAD_CHECKPOINT_PATH) and cfg.CONTINUE_TRAIN
    print(f"Continue Train: {continue_train}")
    if continue_train:
        print(f"CHECKPOINT PATH: {cfg.LOAD_CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(cfg.LOAD_CHECKPOINT_PATH, weights_only=True)['state_dict'], strict=False)
        best_loss = torch.load(cfg.LOAD_CHECKPOINT_PATH, weights_only=True)['best_loss']
        optimizer.load_state_dict(torch.load(cfg.LOAD_CHECKPOINT_PATH, weights_only=True)['optimizer'])
    else:
        best_loss = 1000000000.0

    for name, param in model.state_dict(keep_vars=True).items():
        if param.requires_grad:
            print(f"Parameter: {name},  Param size: {param.numel()}, Requires Grad:{param.requires_grad}")

    writer = SummaryWriter('logs')
    acc_max = 0.0
    acc_max_index = 1

    for epoch in range(cfg.EPOCH_NUM):
        epoch_log = epoch + 1
        model.train() 
        if hasattr(model, 'clip_model'):
            model.clip_model.eval()
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        print(f"Epoch: {epoch_log}")
        print("* epoch     {} * ".format(epoch_log))         
        print(f"Epoch {epoch_log}, Learning Rate(low): {optimizer.param_groups[0]['lr']}, Learning Rate(high): {optimizer.param_groups[-1]['lr']}")
        tbar = tqdm(train_loader, desc='train', ncols=100, disable=False, file=sys.stdout)

        loss_sum = []
        T_cls_sum = []  
        for batch_index, batch in enumerate(tbar):
            with autocast(device_type='cuda'):
                loss_part, T_cls = model(batch)
            loss_sum.append(loss_part)
            T_cls_sum.append(T_cls)
            loss = loss_part[-1]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_sum = [sum(col) for col in zip(*loss_sum)]
        T_cls_sum = [sum(col) for col in zip(*T_cls_sum)]
        scheduler.step()

        batch_index += 1
        for i in range(len(loss_sum)):
            acc = T_cls_sum[i] / train_data.__len__()            
            writer.add_scalar(f'train/loss {part_name[i]}', loss_sum[i] / train_loader.__len__(), epoch)
            writer.add_scalar(f'train/acc {part_name[i]}', acc, epoch)
            print(f"*{part_name[i]} Acc:         {acc} * ")

        with torch.inference_mode():
            model.eval()
            tbar = tqdm(test_loader, desc='test', ncols=100, disable=False, file=sys.stdout)
            loss_sum = []
            T_cls_sum = []
            for batch_index, batch in enumerate(tbar):
                with autocast(device_type='cuda'):
                    loss_part, T_cls = model(batch)
                loss_sum.append(loss_part)
                T_cls_sum.append(T_cls)

            loss_sum = [sum(col) for col in zip(*loss_sum)]
            T_cls_sum = [sum(col) for col in zip(*T_cls_sum)]
            batch_index += 1

            acc_save_flag = False
            for i in range(len(loss_sum)):
                acc = T_cls_sum[i] / test_data.__len__()                
                writer.add_scalar(f'test/loss {part_name[i]}', loss_sum[i] / test_loader.__len__(), epoch)
                writer.add_scalar(f'test/acc {part_name[i]}', acc, epoch)
                print(f"*{part_name[i]} Acc:         {acc} * ")
                if (i == 0) or (i == len(loss_sum) - 1):
                    if acc > acc_max:
                        acc_max = acc
                        acc_max_index = epoch_log
                        acc_save_flag = True
            print(f"*________________* ")
            print(f"*________________* ")
            print(f"*acc_max:     {acc_max} * ")
            print(f"*acc_max_index:     {acc_max_index} * ")
            print(f"*________________* ")
            
            if cfg.SAVE_CHECKPOINT:
                test_loss = loss_sum[0]
                if (best_loss > test_loss) and (epoch_log >= cfg.SAVE_BEGIN_EPOCH):
                    best_loss = test_loss
                    modelname = os.path.join(cfg.SAVE_BEST_LOSS_CHECKPOINT_PATH)
                    print(f"epoch {epoch_log} save best loss dict: loss={test_loss}") 
                    torch.save(
                        {
                            'epoch': epoch_log,
                            'best_loss': best_loss,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, modelname) 

                if (acc_save_flag) and (epoch_log >= cfg.SAVE_BEGIN_EPOCH):
                    modelname = os.path.join(cfg.SAVE_BEST_ACC_CHECKPOINT_PATH)
                    print(f"epoch {epoch_log} save best acc dict: acc={acc_max}") 
                    torch.save(
                        {
                            'epoch': epoch_log,
                            'best_loss': best_loss,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, modelname)   

                if (0 == epoch_log%(cfg.SAVE_GAP)) and (epoch_log >= cfg.SAVE_BEGIN_EPOCH):
                    modelname = os.path.join(cfg.SAVE_CHECKPOINT_PATH)
                    print("epoch {} save dict".format(epoch_log)) 
                    torch.save(
                        {
                            'epoch': epoch_log,
                            'best_loss': best_loss,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, modelname)

if __name__ == '__main__':
    main()

