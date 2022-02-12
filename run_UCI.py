import numpy as np
import os
import argparse
import tools 

import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default = "data", type = str, help = "data directory")
parser.add_argument('-file', default = "out.log", type = str, help = "Output File")
parser.add_argument('-max_total', default = 10000, type = int, help = "Maximum total number of samples")

args = parser.parse_args()

MAX_N_TOT = args.max_total
datadir = args.dir

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
