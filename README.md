# Kernel PCA for Out-of-Distribution Detection: Non-Linear Kernel Selections and Approximations
This is the official PyTorch implementation of the paper: *Kernel PCA for Out-of-Distribution Detection: Non-Linear Kernel Selections and Approximations* ([arxiv](https://arxiv.org/abs/2505.15284)).

This is an **extension study** of our previous work accepted by NeurIPS'24: *Kernel PCA for Out-of-Distribution Detection* ([conference](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html), [arxiv](https://arxiv.org/abs/2402.02949), [code](https://github.com/fanghenshaometeor/ood-kernel-pca)).

*This repo. is being updated...*

If our work benefits your researches, welcome to cite our paper!
```
@inproceedings{fang2024kpcaood,
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

```
@misc{fang2025kpcaood,
title = {Kernel PCA for Out-of-Distribution Detection: Non-Linear Kernel Selections and Approximations}, 
author = {Kun Fang and Qinghua Tao and Mingzhen He and Kexin Lv and Runze Yang and Haibo Hu and Xiaolin Huang and Jie Yang and Longbin Cao},
year = {2025},
eprint = {2505.15284},
archivePrefix = {arXiv},
primaryClass = {cs.LG},
url = {https://arxiv.org/abs/2505.15284}, 
}
```

## KPCA for OoD detection in a nutshell

Out-of-Distribution (OoD) detection is vital for the reliability of deep neural networks, the key of which lies in effectively characterizing the disparities between OoD and  In-Distribution (InD) data.
In this work, such  disparities are exploited through a fresh perspective of *non-linear feature subspace*.
That is, a discriminative non-linear subspace is learned from InD features to capture representative patterns of InD, while informative patterns of OoD features cannot be well captured in such a subspace due to their different distribution. 
Grounded on this perspective, we exploit the deviations of InD and OoD features in such a non-linear subspace for effective OoD detection.
To be specific, we leverage the framework of Kernel Principal Component Analysis (KPCA) to attain the discriminative non-linear subspace and deploy the reconstruction error on such subspace to distinguish InD and OoD data.
Two challenges emerge: *(i)* the learning of an effective non-linear subspace, i.e., the selection of kernel function in KPCA, and *(ii)* the computation of the kernel matrix with large-scale InD data.
For the former, we reveal two vital non-linear patterns that closely relate to the InD-OoD disparity, leading to the establishment of a Cosine-Gaussian kernel for constructing the subspace.
For the latter, we introduce two techniques to approximate the Cosine-Gaussian kernel with significantly cheap computations. 
In particular, our approximation is further tailored by incorporating the InD data confidence, which is demonstrated to promote the learning of discriminative subspaces for OoD data.
Our study presents new insights into the non-linear feature subspace for OoD detection and contributes practical explorations on the associated kernel design and efficient computations, yielding a KPCA detection method with distinctively improved efficacy and efficiency.

### Main differences of this extension study from its [conference version]((https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html)) [1]:
- We supplement more analyses and experiments to support the non-linear kernel selection for OoD detection in Section III.
- A data-dependent Nystr\"om method is employed to build an explicit approximated mapping in Section IV-B, where the sampling strategy in Nystr\"om is intentionally devised based on InD data confidence.
Such a modified data-dependent approximation leads to more discriminative InD and OoD representations and outperforms the data-independent way in [1] with enhanced OoD detection performance and a cheaper computational complexity.
- We supplement a numerical analysis in Section IV-D on the approximation performance of our method in terms of KPCA reconstruction errors on InD and OoD data within this deep learning regime for a comprehensive investigation.
- More experiments are provided in Section V, including comparisons with a broader variety of strong baselines and in-depth analyses on our KPCA detection method to validate its effectiveness.

## Pre-requisite
Prepare in-distribution and out-distribution data sets following the instructions in the [KNN repo](https://github.com/deeplearning-wisc/knn-ood).
Then, modify the data paths in `utils_ood.py` as yours.

For ResNet50 on ImageNet under supervised contrastive learning, download our trained checkpoint and put it as
```
ood-kpca-extension
├── model
├── save
|   └── ImageNet
|       └── R50
|           └── supcon
|               └── supcon-linear.pth
├── ...
```

*The supervised contrast learning R50 checkpoint released in the [KNN repo](https://github.com/deeplearning-wisc/knn-ood) only contains backbone weights and misses the linear layer. We further fine-tune the linear layer on top of the backbone weights following the suggestions in the [supcontrast repo](https://github.com/HobbitLong/SupContrast). Our trained checkpoint is released [here](https://drive.google.com/drive/folders/1-ISbfuEqMZnLpnSud6v2GSl2SNjSiXTJ?usp=sharing).*

## Running
step.1. Run the `feat_extract_largescale.sh` to extract the penultimate layer features.

step.2. 
<!-- - Run the `run_detection.sh` to obtain the detection results where only the KPCA-based reconstruction error serves as the detection score. 
- Run the `run_detection_fusion.sh` to obtain the detection results where the KPCA-based reconstruction error is fused with other detection scores (MSP, Energy, ReAct, BATS). -->

## 

If u have problems about the code or paper, u could contact me (kun.fang@polyu.edu.hk) or raise issues here.

If the code benefits ur researches, welcome to fork and star ⭐ this repo! :)

---
[1] K. Fang, Q. Tao, K. Lv, M. He, X. Huang, and J. Yang. Kernel PCA
for Out-of-Distribution Detection. NeurIPS 2024.