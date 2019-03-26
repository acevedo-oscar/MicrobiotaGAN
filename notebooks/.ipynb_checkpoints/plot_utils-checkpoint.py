import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy 

import seaborn as sns
sns.set(color_codes=True)
import os

from scipy.stats import entropy as DKL


def gan_error(gan_ds, training_ds):
    assert gan_ds.shape == training_ds.shape
    a = gan_ds.mean(axis=0)
    b = training_ds.mean(axis=0)
    return DKL(a,b)

def build_table(data_path, train_ds):

    old_path = os.getcwd()
    os.chdir(data_path)
    cwd = os.getcwd()

    
    directory_list=[x[0].replace(cwd,'').replace('/','') for x in os.walk(cwd,topdown=True)]
    directory_list= directory_list[1:]
    directory_list = sorted(directory_list)
        
    dkl_record = []
    
    id_sample_N = []
    id_repetition = []
    
    # print(len(directory_list))
    # For loop goes here
    for dir_k in range(len(directory_list)):

        numbers_list =(directory_list[dir_k].replace('_data','').replace('_',' ').replace('/',' ')).split(' ')


        os.chdir(directory_list[dir_k])
        ids = [int(numbers_list[k]) for k in range(len(numbers_list))]

        samples_file = 'gan_samples_'+numbers_list[0]+'_'+numbers_list[1] + '.csv'

        fake_samples = pd.read_csv(samples_file,header=None) .values
        indices =  (pd.read_csv('training_indices.csv', header=None) .values).flatten() 
        val = gan_error(fake_samples, train_ds[indices])
        
        dkl_record.append(val)
        id_sample_N.append(ids[0])
        id_repetition.append(ids[1])
        
        os.chdir(cwd)
        
        ## 

    os.chdir(old_path)
    
    plot_table = pd.DataFrame(
    {'DKL': dkl_record,
     'id_sample_N': id_sample_N,
     'id_repetition': id_repetition
    })
    
    return plot_table


def plot_gan_curve(my_dataframe, debug_flag = False):
    mean_y = []
    std_mean = []

    samples_n= np.unique(my_dataframe.id_sample_N.values)

    # Compute mean of observationsß
    for k in range(len(samples_n)):
        a = my_dataframe[my_dataframe.id_sample_N == samples_n[k]]
        #print(a.DKL.values.mean())
        mean_y.append(a.DKL.values.mean())
        
    mean_y = np.array(mean_y)
        
    std_mean = [] # Standard error of the mean

    # Compute the STD
    for k in range(len(samples_n)):
        a = my_dataframe[my_dataframe.id_sample_N == samples_n[k]]
        a = a.DKL.values
        #print(a)
        #print(a.std())
        std_mean.append(a.std()/np.sqrt(len(a))) #/ 

    std_mean =  np.array(std_mean)
    if debug_flag:
        print('Standard error of the mean '+str(std_mean))

    sns.lineplot(samples_n, mean_y, marker = 'o')
    plt.errorbar(samples_n, mean_y, yerr= std_mean, fmt='o')
    #plt.show()


def build_table2(data_path, test_ds):

    old_path = os.getcwd()
    os.chdir(data_path)
    cwd = os.getcwd()

    
    directory_list=[x[0].replace(cwd,'').replace('/','') for x in os.walk(cwd,topdown=True)]
    directory_list= directory_list[1:]
    directory_list = sorted(directory_list)
        
    dkl_record = []
    
    id_sample_N = []
    id_repetition = []
    
    # print(len(directory_list))
    # For loop goes here
    for dir_k in range(len(directory_list)):

        numbers_list =(directory_list[dir_k].replace('_data','').replace('_',' ').replace('/',' ')).split(' ')


        os.chdir(directory_list[dir_k])
        ids = [int(numbers_list[k]) for k in range(len(numbers_list))]

        samples_file = 'gan_samples_'+numbers_list[0]+'_'+numbers_list[1] + '.csv'

        fake_samples = pd.read_csv(samples_file,header=None) .values
        #indices =  (pd.read_csv('training_indices.csv', header=None) .values).flatten() 
        if  fake_samples.shape[0] > test_ds.shape[0] :
            val = gan_error(fake_samples[0:test_ds.shape[0],:], test_ds)
        else :
            val = gan_error(fake_samples, test_ds[0:fake_samples.shape[0],:])

        
        dkl_record.append(val)
        id_sample_N.append(ids[0])
        id_repetition.append(ids[1])
        
        os.chdir(cwd)
        
        ## 

    os.chdir(old_path)
    
    plot_table = pd.DataFrame(
    {'DKL': dkl_record,
     'id_sample_N': id_sample_N,
     'id_repetition': id_repetition
    })
    
    return plot_table