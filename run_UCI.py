#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:27:22 2022

@author: Parisa
"""

'''
source for reading UCI datasets from directories: 
    https://github.com/LeoYu/neural-tangent-kernel-UCI/blob/master/UCI.py
'''
import numpy as np
import os
import argparse
# import math
# import NTK
import tools 

import time
import pickle

path = '/Users/Parisa/Google Drive/python/Codes_Python/NN_gen_resub/UCI_MLP'
os.chdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default = "data", type = str, help = "data directory")
parser.add_argument('-file', default = "result.log", type = str, help = "Output File")
parser.add_argument('-max_tot', default = 5000, type = int, help = "Maximum number of data samples")
parser.add_argument('-max_dep', default = 5, type = int, help = "Maximum number of depth")

args = parser.parse_args()

MAX_N_TOT = args.max_tot
MAX_DEP = args.max_dep
DEP_LIST = list(range(MAX_DEP))
C_LIST = [10.0 ** i for i in range(-2, 5)]
datadir = args.dir


# alg = tools.svm

avg_acc_list = []
outf = open(args.file, "w")
print ("Dataset\tValidation Acc\tTest Acc", file = outf)

ss= time.time()
results = {}
for idx, dataset in enumerate(sorted(os.listdir(datadir))):
    if not os.path.isdir(datadir + "/" + dataset):
        continue
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        continue
    dic = dict()
    for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test
    
    if n_tot > MAX_N_TOT or n_test > 0:
        print (str(dataset) + '\t0\t0', file = outf)
        continue
    
    print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)
    
    # load data
    f = open("data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
    print(dataset)
    
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))
    
   #  # calculate NTK
   # Ks = NTK.kernel_value_batch(X, MAX_DEP)
       
   # load training and validation set
    fold = list(map(lambda x: list(map(int, x.split())), open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]
    
    d = X.shape[1]
    n = len(train_fold)
    results[dataset] = tools.run_exp(xtr=X[train_fold],
                                     ytr=y[train_fold],
                                     xts=X[val_fold],
                                     yts=y[val_fold],
                                     batch_size_list=[int(n/10)], 
                                     dropout_rate_list=[0.0],
                                     width_list=[10, 20, 30, 40]+[i*d for i in [2, 4, 6, 8]],
                                     depth_list=list(range(1, 6)),
                                     n_mc_sample=100)
    
ee = time.time()
    
print('elapsed time = ', (ee-ss)/60, 'minutes')

pickle.dump(results, open('results.pickle', 'wb'))

# xtr, ytr = X[train_fold], y[train_fold] 
# xts, yts = X[val_fold], y[val_fold]   
# width_list=[i*d for i in [3, 6, 9, 12]]
# depth_list=list(range(1, 6))
# n_mc_sample=100    





    
    
    
#     best_acc = 0.0
#     best_value = 0
#     best_dep = 0
#     best_ker = 0

#     # enumerate kenerls and cost values to find the best hyperparameters
#     for dep in DEP_LIST:
#         for fix_dep in range(dep + 1):
#             K = Ks[dep][fix_dep]
#             for value in C_LIST:
#                 acc = alg(K[train_fold][:, train_fold], K[val_fold][:, train_fold], y[train_fold], y[val_fold], value, c)
#                 if acc > best_acc:
#                     best_acc = acc
#                     best_value = value
#                     best_dep = dep
#                     best_fix = fix_dep
    
#     K = Ks[best_dep][best_fix]
    
#     print ("best acc:", best_acc, "\tC:", best_value, "\tdep:", best_dep, "\tfix:", best_fix)
    
#     # 4-fold cross-validating
#     avg_acc = 0.0
#     fold = list(map(lambda x: list(map(int, x.split())), open("data/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
#     for repeat in range(4):
#         train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]
#         acc = alg(K[train_fold][:, train_fold], K[test_fold][:, train_fold], y[train_fold], y[test_fold], best_value, c)
#         avg_acc += 0.25 * acc
        
#     print ("acc:", avg_acc, "\n")
#     print (str(dataset) + '\t' + str(best_acc * 100) + '\t' + str(avg_acc * 100), file = outf)
#     avg_acc_list.append(avg_acc)

# print ("avg_acc:", np.mean(avg_acc_list) * 100)
# outf.close()


