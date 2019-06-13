import pandas as pd
import numpy as np
import csv
from timeit import default_timer as timer

partiton_n = pd.read_csv('random_amounts.csv', header=None).values.flatten()
print(partiton_n)

##
import os
cwd = os.getcwd()
print(cwd)
os.chdir("../../")
##

print("Loading Train func")
from train_dirichlet_interface import train_gan
print("Finished Loading Train func")

train_ds = pd.read_csv('data/k3_dir/k3_train_set.csv', header=None).values
print(train_ds.shape)
ds_size = train_ds.shape[0]

# train_ds = train_ds[0:ds_size,:]

repetitions = 5
batch_size = 256

print("===> Training with CLR policy <===")
print("===> Training with K3 Dirichlet Dataset <===")

#partiton_n = [300, 400, 500, 700]
print("Give this experiment a name")
experiment_name = "k3_2N_faster_gan" #input()
print(experiment_name)

assert type(experiment_name) == str

print(partiton_n)
print(len(partiton_n))

container_path = 'data/'+ experiment_name 
# Recall that os.mkdir isn't recursive, so it only makes on directoryt at a time
try:
    # Create target Directory
    os.mkdir(container_path)
    print("Directory " , container_path ,  " Created ") 
except FileExistsError:
    print("Directory " , container_path ,  " already exists")

for m in range(len(partiton_n)):

    for k_rep in range(repetitions):

        # partiton_n = np.random.randint(batch_size,ds_size) #At least it has to fit one batch size
        index_list = np.random.randint(0,ds_size, size=partiton_n[m]).tolist()

        print("Calling training function")
        ratio = str(k_rep+1)+"/"+str(repetitions)
        ratio2 = str(m+1)+"/"+str(len(partiton_n))
        telegram_message = " >> Repetition ("+ratio+") of Partiton ("+ratio2+")"
        
        start = timer()

        train_gan(train_ds,  index_list, partiton_n[m], k_rep, telegram_message, experiment_name)
        print("====> Finished repetition "+str(k_rep)+' of partition # '+str(m))

        # repetition, partition, size of partition, time
        info_log = [k_rep+1, m+1, partiton_n[m], timer()-start ]

        with open(cwd+'/Time_log.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(info_log)         

# Snippet to write a partition table
"""
partiton_n = [np.random.randint(batch_size,ds_size) for k in range(10) ] 

df = pd.DataFrame(np.array(partiton_n))

save_name = 'random_amounts.csv'
with open(save_name, 'a') as f:
    df.to_csv(f, header=False, index=False)
"""
