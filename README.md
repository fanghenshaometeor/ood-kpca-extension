# Kernel PCA for Out-of-Distribution Detection via Random Features and Low-Rank Approximation
This is the official PyTorch implementation of the **extension study** of the NeurIPS'24 paper [*Kernel PCA for Out-of-Distribution Detection*](https://papers.nips.cc/paper_files/paper/2024/file/f2543511e5f4d4764857f9ad833a977d-Paper-Conference.pdf).

If our work benefits your researches, welcome to cite our paper!
```
@inproceedings{NEURIPS2024_f2543511,
 author = {Fang, Kun and Tao, Qinghua and Lv, Kexin and He, Mingzhen and Huang, Xiaolin and YANG, JIE},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {134317--134344},
 publisher = {Curran Associates, Inc.},
 title = {Kernel PCA for Out-of-Distribution Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/f2543511e5f4d4764857f9ad833a977d-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```

## KPCA for OoD detection in a nutshell

We highlight the main differences of this extension study from its conference version [1]:
- A data-dependent low-rank Nystr\"om method is introduced to approximate KPCA, where the sampling strategy in Nystr\"om is intentionally devised based on InD data properties.  
Such a data-dependent approximation leads to more discriminative InD and OoD representations and outperforms the data-independent random features in [1] with enhanced OoD detection performance and cheaper computational complexity.
- A numerical analysis on the approximation performance of random features and low-rank approximation in terms of KPCA reconstruction errors on InD and OoD data within this deep learning regime is supplemented for a comprehensive investigation.

## Pre-requisite
Prepare in-distribution and out-distribution data sets following the instructions in the [KNN repo](https://github.com/deeplearning-wisc/knn-ood).
Then, modify the data paths in `utils_ood.py` as yours.

For ResNet50 on ImageNet under supervised contrastive learning, download our model checkpoint and put it as
```
ood-kpca-extension
├── model
├── save
|   └── ImageNet
|       └── R50
|           └── supcon
|               └── supcon.pth
├── ...
```

*The supervised contrast learning R50 checkpoint released in the [KNN repo](https://github.com/deeplearning-wisc/knn-ood) only contains backbone weights and misses the linear layer. We additionally train the linear layer on top of the backbone weights following the suggestions in the [supcontrast repo](https://github.com/HobbitLong/SupContrast).*

## Running
step.1. Run the `feat_extract.sh` to extract the penultimate features.

step.2. 
- Run the `run_detection.sh` to obtain the detection results where only the KPCA-based reconstruction error serves as the detection score. 
- Run the `run_detection_fusion.sh` to obtain the detection results where the KPCA-based reconstruction error is fused with other detection scores (MSP, Energy, ReAct, BATS).

---
[1] K. Fang, Q. Tao, K. Lv, M. He, X. Huang, and J. YANG. Kernel pca
for out-of-distribution detection. NeurIPS 2024.