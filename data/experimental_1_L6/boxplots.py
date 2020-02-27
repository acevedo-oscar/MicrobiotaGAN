import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
sys.path.append("../../src/") 

from error_metrics import gan_error_all_species

ground_truth = pd.read_csv("../experimental/Soil_L6.csv").values


def find_indexes(whole, items):
    n = whole.shape[0]
    m = items.shape[0]
    
    assert n > m
    
    my_list = [ ]
    
    for k in range(m):
        for j in range(n):
            
            if items[k] == whole[j]:
                my_list.append(j)
                
    return my_list
    

def produce_plots(experiment_name:str, gan_samples):   

    epochs = pd.read_csv('_epoch_record.csv', header=None).values

    jsd_record = gan_error_all_species(gan_samples , ground_truth, "JSD", vector_solution=True)

    print(jsd_record.shape)

    # JSD boxplot per species
    ds  =jsd_record.flatten()

    plt.figure(figsize=(18.5, 10.5))
    sns.boxplot(ds)
    mess = "; epoch "+str(epochs[-1])
    plt.title("JSD All Species"+mess)

    # Computing the ouliers
    q1 = np.quantile(jsd_record, 0.25)
    q3 = np.quantile(jsd_record, 0.75)

    outliers = jsd_record > (q3 + 1.5*(q3-q1))

    filtered = jsd_record[outliers]  

    outliers_species= np.array(find_indexes(jsd_record, filtered)) + 1

    outlier_mess = "; Outlier Species: " + str(outliers_species)

    #
    plt.xlabel("STD "+str(np.round(ds.std(),2))+outlier_mess)
    plt.ylabel("Mean "+str(np.round(ds.mean(),2 )))

    plt.savefig('n_jsd_boxplot'+".png", dpi=300)
    plt.close()

    # Writing the species that the error is outliers

    



# Crawl thorugh folders


old_path = os.getcwd()

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
        print("Generating Samples "+percentage )

    numbers_list =(directory_list[dir_k].replace('_data','').replace('_',' ').replace('/',' ')).split(' ')

    os.chdir(directory_list[dir_k])
    ids = [int(numbers_list[k]) for k in range(len(numbers_list))]

    samples_file =numbers_list[0]+'_'+numbers_list[1] 

    path = "gan_samples_" +samples_file+ ".csv"

  
    gan_samples = pd.read_csv(path).values

    produce_plots(samples_file, gan_samples) 

    del gan_samples
    
    os.chdir(cwd)
    
    ## 

os.chdir(old_path)

