DATASET_NAME = 'CUB' #'CUB' 'Aircraft' 'Car' 'Dogs'
STAGE = 1 
BACKBONE = 'vit-b' ##resnet50 vit-b swin-b #efficientnet-b7

# Enhanced PLM settings
USE_PRECOMPUTED_MASKS = True
PART_MASK_PATH = 'datasets/RawData/CUB/coarse_mask_m6.pth'



SAVE_CHECKPOINT = False
LOAD_CHECKPOINT_PATH = f"/CHECKPOINT/{DATASET_NAME}_{BACKBONE}_best_acc.pt" 
SAVE_CHECKPOINT_PATH = "/CHECKPOINT/model_last.pt"
SAVE_BEST_LOSS_CHECKPOINT_PATH = f"/CHECKPOINT/{DATASET_NAME}_{BACKBONE}_best_loss.pt"
SAVE_BEST_ACC_CHECKPOINT_PATH = f"/CHECKPOINT/{DATASET_NAME}_{BACKBONE}_best_acc.pt"
SAVE_GAP = 1
SAVE_BEGIN_EPOCH = 1


LOAD_CACHE = False
CUT_BOX = True

CONTINUE_TRAIN = False  

SEED = 1234
DEVICE = 'cuda'

WARMUP_NUM = 10
EPOCH_NUM = 100

if DATASET_NAME=='Aircraft':
    LR = 1e-5   
    EMBED_DIM = 512
    BS = 16
    WD = 0.01
    # For enhanced PLM with DINOv2+clustering
    PART_NAME = [
        'part0', 'part1', 'part2', 'part3', 
        'part4','part5','part6']
elif DATASET_NAME=='CUB':
    LR = 1e-5
    EMBED_DIM = 768
    BS = 16
    WD = 0.01
    PART_NAME = [
        'part0', 'part1', 'part2', 'part3', 'part4','part5',]
elif DATASET_NAME=='Car':
    LR = 1e-5
    EMBED_DIM = 768
    BS = 16
    WD = 0.01
    PART_NAME = [
        'part0', 'part1', 'part2', 'part3', 
        'part4','part5','part6','part7',]
elif DATASET_NAME=='Dogs':
    LR = 1e-6
    EMBED_DIM = 768
    BS = 16
    WD = 0.01
    PART_NAME = [
        'part0', 'part1', 'part2', 'part3', 
        'part4','part5','part6']
