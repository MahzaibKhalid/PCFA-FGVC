import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchpq
from tqdm.auto import tqdm
from torchvision.transforms import transforms
from PIL import Image
import glob

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def remap_to_contiguous(labels: torch.Tensor):
    """Remap arbitrary cluster IDs to contiguous [0..K-1].
    Works for tensors of any shape; returns remapped tensor and id->new_id dict.
    """
    unique_ids = torch.unique(labels)
    id_list = unique_ids.tolist()
    id_map = {int(old_id): new_id for new_id, old_id in enumerate(id_list)}
    remapped = labels.clone()
    for old_id, new_id in id_map.items():
        remapped[labels == old_id] = new_id
    return remapped, id_map

def main():
    # Configuration
    dataset_name = 'Car'  # Car dataset
    seed = 42
    num_samples = 100  # Number of samples for initial clustering
    num_fg_clusters = 2  # Foreground vs background
    num_part_clusters = 12  # Number of aircraft parts
    
    set_seed(seed)
    
    # Set up paths for Car dataset
    rootdir = f'datasets/RawData/{dataset_name}'
    output_dir = 'autodl-fs/cars'
    os.makedirs(f'{rootdir}/visualizations', exist_ok=True)
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)
    
    # Check if DINOv2 features exist
    feature_file = f'{output_dir}/dinov2_image_feats.pth'
    if not os.path.exists(feature_file):
        print(f"Error: Feature file {feature_file} not found.")
        print("Please run extract_features.py first to generate features.")
        return
    
    # Load DINOv2 features
    print("Loading DINOv2 features...")
    sd = torch.load(feature_file, map_location='cpu')
    print(f"Loaded features with shape: {sd.size()}")
    
    # Create transform for visualization
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224)
    ])
    
    # Step 1: Select random subset for initial clustering
    N = sd.size(0)
    randidx = torch.randperm(N)[:num_samples]
    randsd = sd[randidx].permute(0, 2, 1)  # (num_samples, HW, C)
    print(f"Selected {num_samples} random samples with shape: {randsd.size()}")
    
    # Step 2: Foreground-Background Clustering
    print("\nRunning Foreground-Background clustering...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fg_kmeans = torchpq.clustering.KMeans(n_clusters=num_fg_clusters,
                                         distance="cosine",
                                         verbose=1,
                                         n_redo=5,
                                         max_iter=1000)
    
    # Fit on random subset
    fg_labels = fg_kmeans.fit(randsd.reshape(-1, 768).t().contiguous().to(device)).cpu().reshape(num_samples, -1)
    
    # Print unique labels at level 1 (foreground-background clustering)
    unique_labels_level1 = torch.unique(fg_labels)
    print(f"Unique labels at level 1 (foreground-background): {unique_labels_level1}")
    
    # Visualize foreground-background clustering
    fg_vis_path = f'{output_dir}/visualizations/fg_bg_clustering.png'
    plt.figure(figsize=(20, 20))
    for i in range(min(100, num_samples)):
        plt.subplot(10, 10, i+1)
        plt.imshow(fg_labels[i].reshape(16, 16))  # 16x16 for DINOv2
        plt.axis('off')
    plt.savefig(fg_vis_path)
    plt.close()
    print(f"Saved foreground-background visualization to {fg_vis_path}")
    
    # Determine foreground index (this requires manual inspection)
    # For now, assuming the less common cluster is the foreground
    counts = torch.unique(fg_labels, return_counts=True)[1]
    fg_idx = 0 if counts[0] < counts[1] else 1
    bg_idx = 1 - fg_idx
    print(f"Assuming cluster {fg_idx} is foreground and {bg_idx} is background")
    
    # Step 3: Normalize background and prepare for part clustering
    print("\nNormalizing background...")
    randsd_bgnorm = []
    randsd_nobg = []
    randsd_bgmean = []
    
    for i in range(num_samples):
        # Get mean of background features
        bgnorm_mean = randsd[i][fg_labels[i] == bg_idx].mean(dim=0, keepdim=True)
        
        # Create background mask
        if fg_idx == 0:
            bg_mask = fg_labels[i]
        else:
            bg_mask = 1 - fg_labels[i]
            
        bg_mask = bg_mask.unsqueeze(1)
        
        # Normalize background
        bgnorm = (randsd[i] * (1 - bg_mask)) + (bgnorm_mean * bg_mask)
        
        # Store results
        randsd_bgnorm.append(bgnorm)
        randsd_nobg.append(randsd[i] * (1 - bg_mask) + (-1 * bg_mask))  # Set bg to -1
        randsd_bgmean.append(bgnorm_mean)
        
    randsd_bgnorm = torch.stack(randsd_bgnorm, dim=0)
    randsd_nobg = torch.stack(randsd_nobg, dim=0)
    randsd_bgmean = torch.cat(randsd_bgmean, dim=0)
    
    # Step 4: Part Clustering
    print("\nRunning Part Clustering...")
    set_seed(seed)
    
    coarse_kmeans = torchpq.clustering.KMeans(n_clusters=num_part_clusters,
                                             distance="cosine",
                                             verbose=1,
                                             n_redo=5,
                                             max_iter=1000)
    
    coarse_labels = coarse_kmeans.fit(randsd_nobg.reshape(-1, 768).t().contiguous().to(device)).cpu().reshape(num_samples, -1)
    
    # Print unique labels at level 2 (part clustering)
    unique_labels_level2 = torch.unique(coarse_labels)
    print(f"Unique labels at level 2 (part clustering): {unique_labels_level2}")
    
    # Analyze cluster distribution
    print("\nCluster distribution analysis:")
    for cluster_id in unique_labels_level2:
        count = (coarse_labels == cluster_id).sum().item()
        percentage = (count / coarse_labels.numel()) * 100
        print(f"  Cluster {cluster_id}: {count} pixels ({percentage:.2f}%)")
    
    # Analyze per-image cluster usage
    print("\nPer-image cluster usage analysis:")
    cluster_usage = {}
    for i in range(num_samples):
        img_clusters = torch.unique(coarse_labels[i])
        num_clusters = len(img_clusters)
        if num_clusters not in cluster_usage:
            cluster_usage[num_clusters] = 0
        cluster_usage[num_clusters] += 1
    
    for num_clusters, count in sorted(cluster_usage.items()):
        percentage = (count / num_samples) * 100
        print(f"  {num_clusters} clusters: {count} images ({percentage:.1f}%)")
    
    # Remap subset coarse labels to contiguous IDs for visualization
    coarse_labels_remap, subset_id_map = remap_to_contiguous(coarse_labels)
    print(f"Remapped subset cluster IDs to contiguous range [0..{len(subset_id_map)-1}]")

    # Visualize part clustering
    part_vis_path = f'{output_dir}/visualizations/part_clustering.png'
    plt.figure(figsize=(20, 20))
    for i in range(min(100, num_samples)):
        plt.subplot(10, 10, i+1)
        plt.imshow(coarse_labels_remap[i].reshape(16, 16))  # 16x16 for DINOv2
        plt.axis('off')
    plt.savefig(part_vis_path)
    plt.close()
    print(f"Saved part clustering visualization to {part_vis_path}")
    
    # Step 5: Apply clustering to all images
    print("\nApplying clustering to all images...")
    
    # Apply foreground-background clustering
    sd_inp = sd.permute(0, 2, 1)  # (N, HW, C)
    sd_fg_labels = []
    
    bs = 1000  # Processing batch size
    for bidx in tqdm(range(N // bs + 1), desc="FG-BG Clustering"):
        if bidx * bs >= N:
            break
            
        start_bidx = bidx * bs
        end_bidx = min((bidx + 1) * bs, N)
        
        if start_bidx == end_bidx:
            continue
            
        batch_fg_labels = fg_kmeans.predict(
            sd_inp[start_bidx:end_bidx].reshape(-1, 768).t().contiguous().to(device)
        ).cpu().reshape(end_bidx - start_bidx, -1)
        
        sd_fg_labels.append(batch_fg_labels)
        
    sd_fg_labels = torch.cat(sd_fg_labels, dim=0)
    
    # Normalize background for all images
    sd_bgnorm = []
    sd_nobg = []
    sd_bgmean = []
    
    for i in tqdm(range(N), desc="Background Normalization"):
        # Get mean of background features
        bgnorm_mean = sd_inp[i][sd_fg_labels[i] == bg_idx].mean(dim=0, keepdim=True)
        
        # Create background mask
        if fg_idx == 0:
            bg_mask = sd_fg_labels[i]
        else:
            bg_mask = 1 - sd_fg_labels[i]
            
        bg_mask = bg_mask.unsqueeze(1)
        
        # Normalize background
        bgnorm = (sd_inp[i] * (1 - bg_mask)) + (bgnorm_mean * bg_mask)
        
        # Store results
        sd_bgnorm.append(bgnorm)
        sd_nobg.append(sd_inp[i] * (1 - bg_mask) + (-1 * bg_mask))  # Set bg to -1
        sd_bgmean.append(bgnorm_mean)
        
    sd_bgnorm = torch.stack(sd_bgnorm, dim=0)
    sd_nobg = torch.stack(sd_nobg, dim=0)
    sd_bgmean = torch.cat(sd_bgmean, dim=0)
    
    # Apply part clustering to all images
    sd_coarse_labels = []
    
    for bidx in tqdm(range(N // bs + 1), desc="Part Clustering"):
        if bidx * bs >= N:
            break
            
        start_bidx = bidx * bs
        end_bidx = min((bidx + 1) * bs, N)
        
        if start_bidx == end_bidx:
            continue
            
        batch_coarse_labels = coarse_kmeans.predict(
            sd_nobg[start_bidx:end_bidx].reshape(-1, 768).t().contiguous().to(device)
        ).cpu().reshape(end_bidx - start_bidx, -1)
        
        sd_coarse_labels.append(batch_coarse_labels)
        
    sd_coarse_labels = torch.cat(sd_coarse_labels, dim=0)

    # Remap full coarse labels to contiguous IDs [0..K_actual-1]
    sd_coarse_labels_remap, full_id_map = remap_to_contiguous(sd_coarse_labels)
    K_actual = len(full_id_map)
    print(f"Remapped full dataset cluster IDs to contiguous range [0..{K_actual-1}] (K_actual={K_actual})")
    
    # Visualize part clustering for first 100 images
    full_part_vis_path = f'{output_dir}/visualizations/full_part_clustering.png'
    plt.figure(figsize=(20, 20))
    for i in range(min(100, N)):
        plt.subplot(10, 10, i+1)
        coarse_mask = sd_coarse_labels_remap[i].reshape(16, 16)  # 16x16 for DINOv2
        plt.imshow(coarse_mask)
        plt.axis('off')
    plt.savefig(full_part_vis_path)
    plt.close()
    print(f"Saved full part clustering visualization to {full_part_vis_path}")
    
    # Save part masks
    part_mask_path = f'{output_dir}/coarse_mask_m{K_actual}.pth'
    torch.save(sd_coarse_labels_remap.reshape(N, 16, 16).long(), part_mask_path)  # 16x16 for DINOv2
    print(f"Saved part masks to {part_mask_path}")
    
    # Step 6: Extract mean features for each part
    print("\nExtracting mean features for each part...")
    sd_fgmean = []
    
    for i in tqdm(range(N), desc="Part Mean Features"):
        mean_feats = []
        for m in range(K_actual):
            coarse_mask = sd_coarse_labels_remap[i] == m
            if coarse_mask.sum().item() == 0:
                m_mean_feats = torch.zeros(1, 768)
            else:
                m_mean_feats = sd_inp[i][coarse_mask].mean(dim=0, keepdim=True)
            
            mean_feats.append(m_mean_feats)
        
        mean_feats = torch.cat(mean_feats, dim=0)
        sd_fgmean.append(mean_feats)
        
    sd_fgmean = torch.stack(sd_fgmean, dim=0)
    
    # Save mean features
    mean_feat_path = f'{output_dir}/part_mean_features_m{K_actual}.pth'
    torch.save(sd_fgmean, mean_feat_path)
    print(f"Saved part mean features to {mean_feat_path}")
    
    # Final cluster analysis for all images
    print("\nFinal cluster analysis for all images:")
    all_unique_labels = torch.unique(sd_coarse_labels_remap)
    print(f"Total unique clusters found: {len(all_unique_labels)}")
    
    # Overall cluster distribution
    print("\nOverall cluster distribution:")
    for cluster_id in all_unique_labels:
        count = (sd_coarse_labels_remap == cluster_id).sum().item()
        percentage = (count / sd_coarse_labels_remap.numel()) * 100
        print(f"  Cluster {cluster_id}: {count} pixels ({percentage:.2f}%)")
    
    # Per-image cluster usage for all images
    print("\nPer-image cluster usage (all images):")
    cluster_usage_all = {}
    for i in range(N):
        img_clusters = torch.unique(sd_coarse_labels_remap[i])
        num_clusters = len(img_clusters)
        if num_clusters not in cluster_usage_all:
            cluster_usage_all[num_clusters] = 0
        cluster_usage_all[num_clusters] += 1
    
    for num_clusters, count in sorted(cluster_usage_all.items()):
        percentage = (count / N) * 100
        print(f"  {num_clusters} clusters: {count} images ({percentage:.1f}%)")
    
    print("\nClustering completed successfully!")

if __name__ == "__main__":
    main() 
    
    