## Generalized Resubstitution Error Estimation
The theoretical framework for error estimators are explained in our paper [Generalized Resubstitution for Classification Error Estimation](https://arxiv.org/pdf/2110.12285.pdf) 

### Running experiment with synthetic data:
coming soon...

### Running experiment on UCI datasets:
Download the datasets used in paper "Do we need hundreds of classifiers to solve real world classification problems?" from http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz or, alternatively, run 
```
bash setup.sh
```
To run experiments:
```
python run_UCI.py -max_total N -out output_file
```
*options:* <br>
-max_total N: skip the datasets with total of samples larger than N. <br>
-out output_file: the output file.

### Running experiment on MNIST datasets:
