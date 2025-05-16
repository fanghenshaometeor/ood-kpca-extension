from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import sys
import time
import argparse
import re

from utils import setup_seed
from utils import get_model
from utils import Logger
from utils import AverageMeter, accuracy

from utils_ood import make_id_ood



# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Evaluation Energy on training samples')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='./data/CIFAR10/',help='data directory')
parser.add_argument('--logs_dir',type=str,default='./logs/',help='logs directory')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model_path',type=str,default=None,help='saved model path')
parser.add_argument('--supcon',action='store_true',help='extract features from supcon models')
# -------- hyper param. --------
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--seed',type=int,default=0,help='random seeds')
parser.add_argument('--num_classes',type=int,default=10,help='num of classes')
parser.add_argument('--batch_size',type=int,default=128,help='batch size for training (default: 256)')    
# -------- ood param. --------
parser.add_argument('--score', choices=['MSP', 'ReAct', 'Energy', 'BATS', 'GradNorm','RankFeat'], default='MSP')
parser.add_argument('--in_data', choices=['CIFAR10', 'ImageNet'], default='CIFAR10')
parser.add_argument('--in_datadir', type=str, help='in data dir')
parser.add_argument('--out_data', choices=['SVHN','LSUN','iSUN','Texture','places365','iNaturalist','SUN','Places'], default='SVHN')
parser.add_argument('--out_datadir', type=str, help='out data dir')
# --------
parser.add_argument('--temperature_energy', default=1, type=float, help='temperature scaling for energy')
parser.add_argument('--temperature_react', default=1, type=float, help='temperature scaling for React')
parser.add_argument('--threshold_react', default=1, type=float, help='threshold of react')
args = parser.parse_args()

# ======== log writer init. ========
args.dataset = args.in_data
log_file='supcon-%s'%args.score if args.supcon else 'ce-%s'%args.score
if not os.path.exists(os.path.join(args.logs_dir,args.dataset,args.arch,'eval')):
    os.makedirs(os.path.join(args.logs_dir,args.dataset,args.arch,'eval'))
args.logs_path = os.path.join(args.logs_dir,args.dataset,args.arch,'eval',log_file+'-train.log')
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(args.seed)
    
    # ======== get data set =============
    _, in_loader_train, _ = make_id_ood(args)
    print('-------- DATA INFOMATION --------')
    print('---- in-data : '+args.in_data)
    print('---- out-data: '+args.out_data)

    # ======== load network ========
    net = get_model(args).cuda()
    # ----
    if args.in_data == 'CIFAR10':
        checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint['state_dict'])
    if args.in_data == 'ImageNet' and args.supcon:
        checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
        state_dict_model = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
        state_dict_linear = {str.replace(k, 'module.fc.', ''): v for k, v in checkpoint['linear'].items()}
        net.load_state_dict(state_dict_model, strict=False)
        net.fc.load_state_dict(state_dict_linear)
    net.eval()
    print('-------- MODEL INFORMATION --------')
    print('---- arch.: '+args.arch)
    print('---- inf. seed.: '+str(args.seed))
    if args.model_path == None:
        print('---- saved path: Pre-trained ckpt.')
    else:
        print('---- saved path: '+args.model_path)

    # ======== evaluation on Ood ========
    print('Running %s...'%args.score)
    start_time = time.time()
    energy_score, acc_train = val(net, in_loader_train)
    energy_file = 'supcon-%s-train'%args.score if args.supcon else 'ce-%s-train'%args.score
    duration = time.time() - start_time
    np.save(os.path.join(args.logs_dir,args.dataset,args.arch,'eval',energy_file), energy_score)


    print("Finished. Total running time: {}".format(duration))
    print()

    print("energy_score.shape: ", energy_score.shape)
    print("energy_score[0:50]: ", energy_score[0:50])
    print("Training set acc. : ", acc_train)

    return

def val(net, in_loader_train):
    net.eval()
    acc = AverageMeter()
    confs = []
    for b, (x, y) in enumerate(in_loader_train):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            # compute output, measure accuracy and record loss.
            if args.score == 'Energy':
                logits = net(x)
            elif args.score == 'ReAct':
                logits = net.forward_threshold(x, threshold=args.threshold_react)
            else:
                assert False, "Unknown score: {}".format(args.score)

            prec1 = accuracy(logits.data, y)[0]
            acc.update(prec1.item(),x.size(0))

            conf = args.temperature_energy * torch.logsumexp(logits / args.temperature_energy, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs), acc.avg




# ======== startpoint
if __name__ == '__main__':
    main()