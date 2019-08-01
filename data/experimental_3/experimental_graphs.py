import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
    

def produce_plots(experiment_name:str):   

    epochs = pd.read_csv('_epoch_record.csv', header=None).values
    gen_loss = pd.read_csv('_gen_loss.csv', header=None).values
    disc_loss = pd.read_csv('_disc_loss.csv', header=None).values

    training_details = pd.read_csv('_training.csv', )

    n_samples = str(int(training_details.iloc[0][5]))
    # print(n_samples)

    
    tag = '('+n_samples+' samples)'

 
    pre_tag = experiment_name
    pre_tag = '; ['+pre_tag+']'
    # generator

    plt.figure(figsize=(18.5, 10.5))
    plt.plot(epochs,gen_loss)
    plt.title("Generator Loss "+tag+pre_tag)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig('generator_loss.png', dpi=300)
    plt.close()


    # discriminator
    plt.figure(figsize=(18.5, 10.5))

    plt.plot(epochs,disc_loss)
    plt.title("Discriminator Loss "+ tag+pre_tag)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig('discriminator_loss.png', dpi=300)
    plt.close()

    # both
    plt.figure(figsize=(18.5, 10.5))

    plt.plot(epochs,disc_loss, label='Discriminator')
    plt.plot(epochs,gen_loss, label='Generator')
    plt.legend()

    plt.title("GAN Loss " +tag+pre_tag)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.savefig('gan_loss.png', dpi=300)

    plt.close()

    # Plotting the JSD story

    jsd_record = pd.read_csv('_jsd_error.csv', header=None).values
    print(jsd_record.shape)

    plt.figure(figsize=(18.5, 10.5))

    plt.plot(jsd_record[:,0],jsd_record[:,1])
    plt.title("JSD "+ tag+pre_tag)
    plt.xlabel("Epochs")
    plt.ylabel("JSD")

    plt.savefig('JSD_story.png', dpi=300)
    plt.close()

    # JSD boxplot per species

    jsd_record = pd.read_csv("_jsd_vector.csv").values

    # n_spec = jsd_record.shape[1]

    # for k in range(n_spec):

    ds  =jsd_record[-1,:]   

    plt.figure(figsize=(18.5, 10.5))
    sns.boxplot(ds)
    mess = "; epoch "+str(epochs[-1])
    plt.title("JSD All Species"+mess)

    
    # Computing the ouliers

    q1 = np.quantile(ds, 0.25)
    q3 = np.quantile(ds, 0.75)

    outliers = ds > (q3 + 1.5*(q3-q1))

    filtered = ds[outliers]  

    outliers_species= np.array(find_indexes(ds, filtered)) + 1

    outlier_mess = "; Outlier Species: " + str(outliers_species)

    #

    plt.xlabel("STD "+str(np.round(ds.std(),2))+outlier_mess)
    plt.ylabel("Mean "+str(np.round(ds.mean(),2 )))

    plt.savefig('jsd_boxplot'+".png", dpi=300)
    plt.close()



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

    produce_plots(samples_file) 
    
    os.chdir(cwd)
    
    ## 

os.chdir(old_path)

