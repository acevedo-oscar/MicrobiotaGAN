import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy 

import seaborn as sns
sns.set(color_codes=True)
import os

# from scipy.stats import entropy as distance

from scipy.stats import entropy as DKL
from scipy.spatial.distance import jensenshannon as JSD 


def data_probs(ds,bins_partition):
    hist, bin_edges = np.histogram(ds,bins=bins_partition, density=True)
    probs = hist * np.diff(bin_edges)
    return probs

def gan_error(gan_ds, true_ds, error_function):
    """
    Computes the DKL or JSD for 1D data
    """

    assert gan_ds.ndim == true_ds.ndim
    assert gan_ds.ndim == 1

    partitions= np.linspace(true_ds.min(), true_ds.max(),num=100)

    real_distribution = data_probs(true_ds, partitions)
    estimated_distribution  = data_probs(gan_ds, partitions)

    if error_function == "JSD":
        return JSD(estimated_distribution, real_distribution)


    if error_function == "DKL":
        return DKL(estimated_distribution, real_distribution)
    else:
        print("Invalid error functions")

def gan_error_all_species(gan_ds, true_ds, error_function = "JSD"):
    assert gan_ds.shape[1] == true_ds.shape[1]

    N = gan_ds.shape[1]

    scores  = [gan_error(gan_ds[:, k], true_ds[:,k], error_function) for k in range(N) ]

    return np.mean(scores)

def build_table(data_path, train_ds, error_function= "DKL"):

    old_path = os.getcwd()
    os.chdir(data_path)
    cwd = os.getcwd()

    
    directory_list=[x[0].replace(cwd,'').replace('/','') for x in os.walk(cwd,topdown=True)]
    directory_list= directory_list[1:]
    directory_list = sorted(directory_list)
    
    #print(directory_list)
        
    distance_record = []
    
    id_sample_N = []
    id_repetition = []
    
    # print(len(directory_list))
    # For loop goes here
    for dir_k in range(len(directory_list)):

        if dir_k%5 ==0:
            percentage = str(np.round(100*dir_k /len(directory_list),2))+"%"
            print("Table Building progress "+percentage )

        numbers_list =(directory_list[dir_k].replace('_data','').replace('_',' ').replace('/',' ')).split(' ')

        os.chdir(directory_list[dir_k])
        ids = [int(numbers_list[k]) for k in range(len(numbers_list))]

        samples_file = 'gan_samples_'+numbers_list[0]+'_'+numbers_list[1] + '.csv'

        fake_samples = pd.read_csv(samples_file,header=None) .values
        indices =  (pd.read_csv('training_indices.csv', header=None) .values).flatten() 
        val = gan_error_all_species(fake_samples, train_ds[indices])
        
        distance_record.append(val)
        id_sample_N.append(ids[0])
        id_repetition.append(ids[1])
        
        os.chdir(cwd)
        
        ## 

    os.chdir(old_path)
    
    plot_table = pd.DataFrame(
    {'distance': distance_record,
     'id_sample_N': id_sample_N,
     'id_repetition': id_repetition
    })
    
    return plot_table



def build_table2(data_path, test_ds):

    old_path = os.getcwd()
    os.chdir(data_path)
    cwd = os.getcwd()

    
    directory_list=[x[0].replace(cwd,'').replace('/','') for x in os.walk(cwd,topdown=True)]
    directory_list= directory_list[1:]
    directory_list = sorted(directory_list)
        
    distance_record = []
    
    id_sample_N = []
    id_repetition = []
    
    # print(len(directory_list))
    # For loop goes here
    for dir_k in range(len(directory_list)):
        
        if dir_k%5 ==0:
            percentage = str(np.round(100*dir_k /len(directory_list),2))+"%"
            print("Table Building progress "+percentage )


        numbers_list =(directory_list[dir_k].replace('_data','').replace('_',' ').replace('/',' ')).split(' ')


        os.chdir(directory_list[dir_k])
        ids = [int(numbers_list[k]) for k in range(len(numbers_list))]

        samples_file = 'gan_samples_'+numbers_list[0]+'_'+numbers_list[1] + '.csv'
        # print(samples_file)

        fake_samples = pd.read_csv(samples_file,header=None) .values
        #indices =  (pd.read_csv('training_indices.csv', header=None) .values).flatten() 
        if  fake_samples.shape[0] > test_ds.shape[0] :
            val = gan_error_all_species(fake_samples[0:test_ds.shape[0],:], test_ds)
        else :
            val = gan_error_all_species(fake_samples, test_ds[0:fake_samples.shape[0],:])

        
        distance_record.append(val)
        id_sample_N.append(ids[0])
        id_repetition.append(ids[1])
        
        os.chdir(cwd)
        
        ## 

    os.chdir(old_path)
    
    plot_table = pd.DataFrame(
    {'distance': distance_record,
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
        #print(a.distance.values.mean())
        mean_y.append(a.distance.values.mean())
        
    mean_y = np.array(mean_y)
        
    std_mean = [] # Standard error of the mean

    # Compute the STD
    for k in range(len(samples_n)):
        a = my_dataframe[my_dataframe.id_sample_N == samples_n[k]]
        a = a.distance.values
        #print(a)
        #print(a.std())
        std_mean.append(a.std()/np.sqrt(len(a))) #/ 

    std_mean =  np.array(std_mean)
    if debug_flag:
        print('Standard error of the mean '+str(std_mean))

    sns.lineplot(samples_n, mean_y, marker = 'o')
    plt.errorbar(samples_n, mean_y, yerr= std_mean, fmt='o')
    #plt.show()


def plot_abundance_histogram(estimated_distribution, true_distribution, k_esp):
    
    particiones = np.linspace(true_distribution.min(), true_distribution.max(),num=100)

    sns.distplot(true_distribution, bins=particiones, kde=False,label='Real') 
    sns.distplot(estimated_distribution, bins=particiones, kde=False, label='GAN') 
    plt.title("Especie "+str(k_esp))
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

def balance_ds(ds1,ds2):

    if len(ds1) > len(ds2):
        return ds1[0:len(ds2)], ds2
    else: 
        return ds1, ds2[0:len(ds1)]


