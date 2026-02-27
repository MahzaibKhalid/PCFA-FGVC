import os
import shutil

import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from loguru import logger

from model.PACF import PACF_model
from utils.dataset import Aircraft_Dataset, CUB_Dataset, Car_Dataset, Dogs_Dataset, PACF_Collate
import torch.nn as nn
from torch.amp import GradScaler, autocast
import warnings
import config as cfg
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR,CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

@logger.catch 
def main():
    device = torch.device(cfg.DEVICE)
    if cfg.DATASET_NAME=='Aircraft':
        test_data = Aircraft_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    elif cfg.DATASET_NAME=='CUB':
        test_data = CUB_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    elif cfg.DATASET_NAME=='Car':
        test_data = Car_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    elif cfg.DATASET_NAME=='Dogs':
        test_data = Dogs_Dataset(pattern="test", device=device, load_cache=cfg.LOAD_CACHE, cut_box=cfg.CUT_BOX)
    else:    
        warnings.warn("Undefined dataset", ImportWarning)
        return
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=cfg.BS,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=False,
                                            collate_fn=PACF_Collate
                                            )
    part_name = cfg.PART_NAME
    model = PACF_model(stage=cfg.STAGE, embed_dim=cfg.EMBED_DIM, backbone=cfg.BACKBONE, dataset_name=cfg.DATASET_NAME, part_name=part_name, device=device, use_precomputed_masks=cfg.USE_PRECOMPUTED_MASKS, part_mask_path=cfg.PART_MASK_PATH)
    part_name.insert(0, "Coarse")
    part_name.append("Sum")
    max_length = max(len(s) for s in part_name)
    part_name = [s.ljust(max_length) for s in part_name]

    model.load_state_dict(torch.load(cfg.LOAD_CHECKPOINT_PATH, weights_only=True)['state_dict'], strict=False)      
    with torch.inference_mode():
        model.eval()
        tbar = tqdm(test_loader, desc='test', ncols=100)
        loss_sum = []
        T_cls_sum = []
        for batch_index, batch in enumerate(tbar):
            with autocast(device_type='cuda'):
                loss_part, T_cls = model(batch)
            loss_sum.append(loss_part)
            T_cls_sum.append(T_cls)

        loss_sum = [sum(col) for col in zip(*loss_sum)]
        T_cls_sum = [sum(col) for col in zip(*T_cls_sum)]
        for i in range(len(loss_sum)):
            acc = T_cls_sum[i] / test_data.__len__()                
            logger.info(f"*{part_name[i]} Acc:         {acc} * ")


if __name__ == '__main__':
    main()
