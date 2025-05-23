import os
import sys

import time
import torch
import numpy as np
import scipy

import sys
import argparse

from utils import setup_seed
from utils import Logger
from utils import get_model

from utils_ood import make_id_ood
from utils_ood import val_ood_fuse

from metrics import print_all_results
from sklearn.kernel_approximation import pairwise_kernels

# ======== options ==============
parser = argparse.ArgumentParser(description="Kernel PCA for OoD Detection")
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--cache_dir',type=str,default='./cache/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,default='./save/',help='saved model path')
# -------- hyper param. --------
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--num_classes',type=int,default=10,help='num of classes')
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')
parser.add_argument('--supcon',action='store_true',help='extract features from supcon models')
# -------- ood param. --------
parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','kNN'], default='MSP')
parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='ImageNet')
parser.add_argument('--in_datadir', type=str, help='in data dir')
parser.add_argument('--out_data', type=str)
parser.add_argument('--out_datadir', type=str, help='out data dir')
parser.add_argument('--out_datasets',default=['Texture','iNaturalist','SUN','Places'], nargs="*", type=str)
parser.add_argument('--temperature_energy', default=1, type=float, help='temperature scaling for energy')
parser.add_argument('--temperature_react', default=1, type=float, help='temperature scaling for React')
parser.add_argument('--threshold_react', default=1, type=float, help='threshold of react')
# -------- Kernel PCA param. --------
parser.add_argument('--approx', choices=['RFF', 'NYS'], default='NYS')
parser.add_argument('--exp_var_ratio', type=float, help='explained variance ratio to choose q')
parser.add_argument('--gamma', type=float, help='variance of the gaussian kernel')
parser.add_argument('--M', type=int, help='mapped dimension of RFFs, or, num. of landmarks in Nystrom')
args = parser.parse_args()

args.dataset = args.in_data
args.cache_path = os.path.join(args.cache_dir,args.dataset,args.arch,'ce')
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'fusion')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'fusion'))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'fusion',"{}+ReAct-ood.log".format(args.approx))
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

setup_seed(args.seed)

print()
print("-------- -------- --------")
print("The program starts, {}+ReAct...".format(args.approx))

print()
print("Loading features from {}...".format(args.cache_path))
featdim = {
    'R50': 2048,
    'MNet': 1280,
}[args.arch]
id_train_size = 1281167
id_val_size = 50000

cache_name = os.path.join(args.cache_path, "train_in.mmap")
feat_log = np.memmap(cache_name, dtype=float, mode='r', shape=(id_train_size, featdim))

cache_name = os.path.join(args.cache_path, "val_in.mmap")
feat_log_val = np.memmap(cache_name, dtype=float, mode='r', shape=(id_val_size, featdim))

ood_dataset_size = {
    'iNaturalist':10000,
    'SUN': 10000,
    'Places': 10000,
    'Texture': 5640
}
ood_feat_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = os.path.join(args.cache_path, f"{ood_dataset}_out.mmap")
    ood_feat_log = np.memmap(cache_name, dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], featdim))
    ood_feat_log_all[ood_dataset] = ood_feat_log
print("Features loaded.")
print("Feature dimension = %d"%feat_log.shape[1])

print()
print("Features processed via ReAct: CLIP FEAT, threshold=1.0")
preprocess_react = lambda x: np.clip(a=x, a_min=None, a_max=args.threshold_react)

# -------- such an l2 normalization indicates a non-linear mapping w.r.t. a cosine kernel
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(preprocess_react(x)))

ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)
food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

# -------- using the random Fourier features and Nystrom method to approximate the Gaussian kernel
if args.approx == 'RFF':
    m = ftrain.shape[1]
    gamma, M = args.gamma, args.M
    w = np.sqrt(2*gamma)*np.random.normal(size=(M,m))   # generate M i.i.d. samples from p(w)
    u = 2 * np.pi * np.random.rand(M)
    # ---- RFF mapping
    ftrain = np.sqrt(2/M)*np.cos((ftrain.dot(w.T)+u[np.newaxis,:]))
    ftest = np.sqrt(2/M)*np.cos((ftest.dot(w.T)+u[np.newaxis,:]))
    for ood_dataset, food in food_all.items():
        food_all[ood_dataset] = np.sqrt(2/M)*np.cos((food.dot(w.T)+u[np.newaxis,:]))
    print()
    print("Approximation: random Fourier features")
    print("gamma = %f, M = %d"%(gamma, M))
elif args.approx == 'NYS':
    # ---- load energy scores of training samples w.r.t ReAct
    energy_training = np.load(os.path.join(args.logs_dir,args.dataset,args.arch,'eval', "ce-ReAct-train.npy"))
    # ---- select #ldmk with the smallest-M largest energy scores as the landmark points for Nystrom method
    largest_ind = np.argsort(energy_training)[:args.M]
    # ---- obtain the low-rank approximation mapping of Nystrom
    basis = ftrain[largest_ind]
    basis_kernel = pairwise_kernels(basis, metric='rbf', filter_params=True, gamma=args.gamma)
    U, S, V = scipy.linalg.svd(basis_kernel)
    S = np.maximum(S, 1e-12)
    normalization = np.dot(U / np.sqrt(S), V)
    # ---- low-rank projection
    ftrain = np.dot(pairwise_kernels(ftrain, basis, metric='rbf', filter_params=True, gamma=args.gamma), normalization.T)
    ftest = np.dot(pairwise_kernels(ftest, basis, metric='rbf', filter_params=True, gamma=args.gamma), normalization.T)
    for ood_dataset, food in food_all.items():
        food_all[ood_dataset] = np.dot(pairwise_kernels(food, basis, metric='rbf', filter_params=True, gamma=args.gamma), normalization.T)
    print()
    print("Approximation: Nystrom")
    print("gamma = %f, M = %d"%(args.gamma, args.M))
else:
    assert False, "Unknown Approximation: {}".format(args.approx)

# -------- centralize the mapped features
mu = ftrain.mean(axis=0)
ftrain = ftrain - mu
ftest = ftest - mu
for ood_dataset, food in food_all.items():
    food_all[ood_dataset] = food - mu

# -------- linear PCA
print()
print("Running eigen-decomposition...")
Sigma = ftrain.T.dot(ftrain)
u_full, s, _ = scipy.linalg.svd(Sigma)
# ---- the reduction dimension q is
# ---- selected according to the explained variance ratio
q, s_accuml = -1, np.zeros(ftrain.shape[1])
for i in range(ftrain.shape[1]):
    s_accuml[i] = sum(s[:i]) / sum(s)
    if i > 0 and q < 0:
        if s_accuml[i-1] < args.exp_var_ratio and s_accuml[i] >= args.exp_var_ratio:
            q = i
print("Finished.")
print("explained variance ratio = %f"%args.exp_var_ratio)
print("reduction dimension    q = %d"%q)
print("s_accuml at q-1 = %f"%s_accuml[q-1])
print("s_accuml at q   = %f"%s_accuml[q])
print("s_accuml at q+1 = %f"%s_accuml[q+1])

# -------- calculate the reconstruction error for the fusion
u_q = u_full[:,:q]
reconstruct_in = u_q.dot(u_q.T).dot(ftest.T).T
error_in = np.linalg.norm(ftest-reconstruct_in, ord=2, axis=1)

error_out = {}
for ood_dataset, food in food_all.items():
    reconstruct_ood = u_q.dot(u_q.T).dot(food.T).T 
    error_out[ood_dataset] = np.linalg.norm(food-reconstruct_ood, ord=2, axis=1)
    
# -------- compute the ReAct score for fusion
# ---- load network
net = get_model(args).cuda()
net.eval()

# ---- fusion
all_results = []
for out_data in args.out_datasets:
    print()
    args.out_data = out_data
    in_loader, _, out_loader = make_id_ood(args)
    print('-------- DATA INFOMATION --------')
    print('---- in-data : '+args.in_data)
    print('---- out-data: '+args.out_data)

    fpr95, auroc, aupr_in = val_ood_fuse(net, in_loader, out_loader, error_in, error_out[out_data], args)
    results = dict()
    results['FPR'], results['AUROC'], results['AUIN'] = fpr95, auroc, aupr_in
    all_results.append(results)
print()
print_all_results(all_results, args.out_datasets, '{}+ReAct'.format(args.approx))

print()
print("The program ends...")
print("-------- -------- --------")
print()

