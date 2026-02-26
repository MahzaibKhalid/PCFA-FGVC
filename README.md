# PCFA-FGVC

Official PyTorch implementation of **PCFA** for Fine-Grained Visual Classification (FGVC).

This repository provides training, evaluation, and mask precomputation pipelines for applying PCFA on standard fine-grained datasets such as CUB-200-2011, FGVC-Aircraft, Stanford Cars, and Stanford Dogs.

---

## 1. Overview

PCFA is a part-aware feature aggregation framework for fine-grained visual recognition.

The workflow consists of:

1. Dataset preparation
2. Offline part mask generation (unsupervised)
3. Model training
4. Evaluation

All components required to reproduce the experimental pipeline are included in this repository.

---

## 2. Installation

### Requirements

* Python 3.8+
* PyTorch 2.0+
* CUDA (for GPU training)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

Datasets are **not included** in this repository.

Please download the official datasets from their respective sources:

* CUB-200-2011
* FGVC-Aircraft
* Stanford Cars
* Stanford Dogs

After downloading, organize them as follows:

```
datasets/
├── RawData/
│   ├── CUB/images/
│   ├── Aircraft/images/
│   ├── Car/images/
│   └── Dogs/images/
│
└── SupplementaryData/
    ├── CUB/
    ├── Aircraft/
    ├── Car/
    └── Dogs/
```

Ensure the directory structure matches exactly before training.

---

## 4. Part Mask Generation (Offline Step)

Before training, part masks must be generated once per dataset.

This process:

* Extracts frozen self-supervised features
* Performs unsupervised clustering
* Produces spatial part masks
* Saves masks as a `.pth` file

Example:

```bash
python extract_features.py --dataset CUB
```

The generated mask file will be saved and reused during training and evaluation.

### Important Note

Part masks are generated using all images in the dataset (train, validation, and test splits) in a fully unsupervised manner. No class labels are used during mask construction.

---

## 5. Training

After masks are generated:

```bash
python train.py --dataset CUB --backbone vit_b
```

Training will:

* Load precomputed part masks
* Train the classification model
* Save checkpoints to the output directory

---

## 6. Evaluation

To evaluate a trained model:

```bash
python eval.py --dataset CUB --checkpoint path/to/checkpoint.pth
```

Evaluation will report standard classification metrics.

---

## 7. Reproducibility

* Mask generation is deterministic once computed.
* Backbone used for mask generation remains frozen.
* Random seeds are fixed in configuration files.

For exact reproduction, use the same dataset structure and configuration settings.

---

## 8. Repository Structure

```
models/        → Model architecture components  
utils/         → Utility functions and clustering logic  
train.py       → Training script  
eval.py        → Evaluation script  
extract_features.py → Mask generation script  
```

---

## 9. Citation

If you use this repository in your research, please cite the associated paper.

BibTeX entry will be updated upon publication.

---

## 10. License

This project is released under the MIT License.

