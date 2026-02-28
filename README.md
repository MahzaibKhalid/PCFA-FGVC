[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18787438.svg)](https://doi.org/10.5281/zenodo.18787438)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
# PCFA-FGVC

Official PyTorch implementation of **PCFA** for Fine-Grained Visual Classification (FGVC).

This repository provides code for:

* Part mask generation
* Model training
* Model evaluation

It supports standard fine-grained datasets including CUB-200-2011, FGVC-Aircraft, Stanford Cars, and Stanford Dogs.

---

## 1. Requirements

* Python 3.8+
* PyTorch 2.0+
* CUDA-enabled GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

Datasets are **not included** in this repository.

Please download the official datasets from their respective sources and organize them as follows:

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

Ensure that the directory structure matches exactly before proceeding.

---

## 3. Part Mask Generation
Before training, part masks must be generated once for each dataset.

This step:

* Extracts frozen self-supervised features
* Performs unsupervised clustering
* Generates spatial part masks
* Saves masks as a `.pth` file

Example:

```bash
python extract_features.py
python clustering.py
```

The generated mask file will be saved and reused during training and evaluation.

### Note on Mask Generation

Part masks are generated using all available images (train, validation, and test splits) in a fully unsupervised manner. No class labels are used during this process.

---

## 4. Training

After generating part masks, train the model:

```bash
python train.py 
```

Available backbones include:

* resnet50
* vit-b
* swin-b
* efficientnet-b7
  

Model checkpoints will be saved in the `checkpoints/` directory.

---

## 5. Evaluation

To evaluate a trained model:

```bash
python eval.py
```

Evaluation reports standard classification performance metrics.

---

## 6. Pretrained Model Weights

---

Pretrained model checkpoints for **ViT-B on CUB-200-2011** are publicly available on Zenodo:

🔗 https://zenodo.org/records/18807756  
📌 DOI: https://doi.org/10.5281/zenodo.18787437

This release includes:

- `PCFA_CUB_vit_b_best_acc.pt` — Best accuracy checkpoint  
- `PCFA_CUB_vit_b_best_loss.pt` — Best loss checkpoint  
- `PCFA_CUB_vit_b_model_last.pt` — Final training checkpoint  

These weights correspond to the results reported in the manuscript submitted to *The Visual Computer (Springer Nature)*.

After downloading, place the checkpoints inside:
checkpoints/
Then specify the checkpoint path in the configuration file before evaluation.

## 7. Reproducibility

* Mask generation is performed once and reused.
* The backbone used for mask extraction remains frozen.
* Random seeds are fixed.
* All experiments use the same dataset structure described above.

---

## 8. Citation

If you use this repository in your research, please cite the associated paper.

The BibTeX entry will be updated upon publication.

---

## 9. License

This project is released under the MIT License.

