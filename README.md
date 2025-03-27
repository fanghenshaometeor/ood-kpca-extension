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


## Pre-requisite
Prepare in-distribution and out-distribution data sets following the instructions in the [KNN repo](https://github.com/deeplearning-wisc/knn-ood).
Then, modify the data paths in `utils_ood.py` as yours.

For ResNet50 on ImageNet under supervised contrastive learning, download our model checkpoint and put it as
```
ood-kernel-pca
├── model
├── save
|   └── ImageNet
|       └── R50
|           └── supcon
|               └── supcon.pth
├── ...
```

*The supervised contrast learning R50 checkpoint released in [KNN repo](https://github.com/deeplearning-wisc/knn-ood) only contains backbone weights and misses the linear layer. We additionally train the linear layer on top of the backbone weights.*

## Running
step.1. Run the `feat_extract.sh` to extract the penultimate features.

step.2. 
- Run the `run_detection.sh` to obtain the detection results where only the KPCA-based reconstruction error serves as the detection score. 
- Run the `run_detection_fusion.sh` to obtain the detection results where the KPCA-based reconstruction error is fused with other detection scores (MSP, Energy, ReAct, BATS).