# ---------------------------------------------------------
# Python Utils Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_model_setting(locals_):
    print("Uppercase local vars:")

    all_vars = [(k, v) for (k, v) in locals_.items() if (
        k.isupper() and k != 'T' and k != 'SETTINGS' and k != 'ALL_SETTINGS')]
    all_vars = sorted(all_vars, key=lambda x: x[0])

    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))


_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_iter = [0]


def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush(save_folder):
    prints = []

    for name, vals in _since_last_flush.items():
        sum_ = 0
        keys = vals.keys()
        values = vals.values()
        num_keys = len(list(keys))
        for val in values:
            sum_ += val

        prints.append("{}\t{}".format(name, sum_/num_keys))
        _since_beginning[name].update(vals)

        x_vals = _since_beginning[name].keys()
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(save_folder, name.replace(' ', '_')+'.jpg'))

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

"""
Acevedo's Functions
"""    

from numpy import ndarray as Tensor
from scipy import stats

import numpy as np

def Diagonal_Matrix(input_vector: Tensor, spec: int) -> Tensor:
    return np.multiply(np.identity(spec), input_vector)


def GLV_Model(solution_point: Tensor, A_m: Tensor, R_vec: Tensor) -> Tensor:

    n = A_m.shape[0]
    A = Diagonal_Matrix(solution_point.reshape(1, n), n)

    solution_point = solution_point.reshape(n,1)
    R_vec = R_vec.reshape(n,1)
    
    B = np.matmul(A_m, solution_point)+R_vec

    glv_vec = np.matmul(A, B)

    # We are using norm |v|_1
    return np.linalg.norm(glv_vec, ord=1) / glv_vec.shape[0]

def process_data(sample_table):
    
    #tanh(x) preserves the dynamic range
    sample_table = np.tanh(sample_table)
    
    sample_mean = np.mean(sample_table, axis=0)
    sample_std = np.std(sample_table, axis = 0)
    
    z_vector = stats.zscore(sample_table)
    
    return [z_vector, sample_mean, sample_std]

def inverse_process_data(z_vector, sample_mean, sample_std):
    
    x = np.multiply(z_vector, sample_std) + sample_mean
    
    print(x.shape)
    
    print("positives "+str(np.sum(x >= 1)))
    print("negatives "+str(np.sum(x <= -1)))

    x[x >= 1] = 0.999
    x[x <= 1] = -0.999  


    
    """
    if np.sum(x >= 1) > 0:
        print("Unvalid values for tanh ecountered, x >=1")
        x_hat = x[x >= 1] = 0.999
        x = x_hat    
    
            
    if np.sum(x <= -1) > 0:
        print("Unvalid values for tanh ecountered, x <= -1")
        x_hat = x[x <= 1] = -0.999  
        x = x_hat
    """


          
    
    return np.arctanh(x)
    