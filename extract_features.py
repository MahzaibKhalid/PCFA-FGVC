import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers.utils import constants
from PIL import Image
import glob

from dino import DINO

def load_bboxes(rootdir: str, dataset_name: str):
    """Load a mapping from image relative path to bbox (x,y,w,h).
    Supports CUB format and a generic fallback 'bounding_boxes.txt' with 'relpath x y w h'.
    Returns: dict[str, tuple[float,float,float,float]]
    """
    bbox_map = {}
    # CUB format
    cub_images = os.path.join(rootdir, 'images.txt')
    cub_bboxes = os.path.join(rootdir, 'bounding_boxes.txt')
    if os.path.exists(cub_images) and os.path.exists(cub_bboxes):
        id_to_path = {}
        with open(cub_images, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    rel = parts[1]
                    id_to_path[idx] = rel
        with open(cub_bboxes, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    idx = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    if idx in id_to_path:
                        bbox_map[id_to_path[idx]] = (x, y, w, h)
        return bbox_map
    # Generic fallback: one file with per-line 'relpath x y w h'
    generic = os.path.join(rootdir, 'bounding_boxes.txt')
    if os.path.exists(generic):
        with open(generic, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    rel = parts[0]
                    x, y, w, h = map(float, parts[1:5])
                    bbox_map[rel] = (x, y, w, h)
    # Supplementary path: datasets/SupplementaryData/<dataset>/box.txt
    supp_box = os.path.join('datasets', 'SupplementaryData', dataset_name, 'box.txt')
    if os.path.exists(supp_box):
        with open(supp_box, 'r') as f:
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                # Expect formats like: 'relpath x y w h' or 'relpath x1 y1 x2 y2'
                if len(parts) >= 5:
                    rel = parts[0]
                    a, b, c, d = map(float, parts[1:5])
                    # Heuristic: if c>a and d>b considerably, treat as x2,y2
                    if c > a and d > b and (c - a) > 1 and (d - b) > 1:
                        x, y, w, h = a, b, c - a, d - b
                    else:
                        x, y, w, h = a, b, c, d
                    bbox_map[rel] = (x, y, w, h)
    return bbox_map

class CarsDataset(Dataset):
    def __init__(self, image_paths, transform=None, bbox_map=None, image_root=None):
        self.image_paths = image_paths
        self.transform = transform
        self.bbox_map = bbox_map or {}
        self.image_root = image_root or ''
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        # Crop by bbox if available
        try:
            rel = os.path.relpath(image_path, self.image_root) if self.image_root else os.path.basename(image_path)
            if rel in self.bbox_map:
                x, y, w, h = self.bbox_map[rel]
                W, H = image.size
                x1 = max(0, int(round(x)))
                y1 = max(0, int(round(y)))
                x2 = min(W, int(round(x + w)))
                y2 = min(H, int(round(y + h)))
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
        except Exception:
            pass
        
        if self.transform:
            image = self.transform(image)
        
        return {"pixel_values": image, "index": idx}

def main():
    # Configuration
    MEAN = constants.IMAGENET_DEFAULT_MEAN
    STD = constants.IMAGENET_DEFAULT_STD
    dataset_name = 'Car'
    batch_size = 32
    resize = 256
    crop = 224
    
    # Set up paths
    rootdir = f'autodl-fs/datasets/RawData/{dataset_name}'
    output_dir = 'autodl-fs/cars'
    os.makedirs(rootdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image paths
    image_dir = os.path.join(rootdir, 'images')
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return
    
    image_paths = glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)
    image_paths += glob.glob(os.path.join(image_dir, '**', '*.JPG'), recursive=True)
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # Create dataset and dataloader
    # Load bounding boxes (if available) and use for cropping
    bbox_map = load_bboxes(rootdir, dataset_name)
    if len(bbox_map) == 0:
        print("Warning: No bounding boxes found. Proceeding without bbox cropping.")
    else:
        print(f"Loaded {len(bbox_map)} bounding boxes.")

    dataset = CarsDataset(
        image_paths=image_paths,
        transform=transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        bbox_map=bbox_map,
        image_root=image_dir
    )
    
    # Test the first few items in the dataset to ensure they load properly
    print("\nTesting dataset access:")
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            if isinstance(sample, dict) and "pixel_values" in sample:
                print(f"  Sample {i}: ✓ Successfully loaded image with shape {sample['pixel_values'].shape}")
            else:
                print(f"  Sample {i}: ✗ Unexpected format: {type(sample)}")
        except Exception as e:
            print(f"  Sample {i}: ✗ Error loading: {str(e)}")
    
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    # Initialize model
    model = DINO()
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract features
    image_feats = []
    with tqdm(dataloader, bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
        for i, batch in enumerate(tepoch):
            # Get pixel_values from the batch dictionary
            image = batch["pixel_values"].to(device)
            # Optionally get indices if needed
            indices = batch["index"] if "index" in batch else None
            
            with torch.no_grad():
                output = model.get_feat_maps(image)  # (B, C, H, W)
                
            B, C, H, W = output.size()
            output = output.reshape(B, C, H * W)
            image_feats.append(output.cpu())
    
    # Save features
    image_feats = torch.cat(image_feats, dim=0)  # (N, C, H*W)
    feature_path = f'{output_dir}/dinov2_image_feats.pth'
    print(f"Saving features to {feature_path}, shape: {image_feats.size()}")
    torch.save(image_feats, feature_path)
    
    print(f"Feature extraction complete! Features saved to {feature_path}")
    print(f"You can now run clustering to discover car parts.")

if __name__ == "__main__":
    main() 