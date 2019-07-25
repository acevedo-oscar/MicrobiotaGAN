import pandas as pd
import numpy as np
import csv
from timeit import default_timer as timer

##
import os
cwd = os.getcwd()
print(cwd)
os.chdir("../") #../
##

print("Loading Train func")
from train_experimental_interface import train_gan
print("Finished Loading Train func")

ds_path = 'data/experimental/Soil_L6.csv'
train_ds = pd.read_csv(ds_path, header=None).values
test_set = pd.read_csv(ds_path, header=None).values

print(train_ds.shape)
ds_size = train_ds.shape[0]

# train_ds = train_ds[0:ds_size,:]

repetitions = 5
# batch_size = 2*256

print("Give this experiment a name")
experiment_name = "experimental_1" #input()
print(experiment_name)

assert type(experiment_name) == str
 

container_path = 'data/'+ experiment_name 
# Recall that os.mkdir isn't recursive, so it only makes on directoryt at a time
try:
    # Create target Directory
    os.mkdir(container_path)
    print("Directory " , container_path ,  " Created ") 
except FileExistsError:
    print("Directory " , container_path ,  " already exists")
 
for k_rep in range(repetitions):

    # partiton_n = np.random.randint(batch_size,ds_size) #At least it has to fit one batch size 
    print(k_rep) 
    print("Calling training function")

    start = timer()

    train_gan(train_ds, test_set, k_rep, experiment_name)
    print("====> Finished repetition "+str(k_rep))

    # repetition, partition, size of partition, time
    info_log = [k_rep+1, timer()-start ]

    with open(cwd+'/Time_log.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(info_log)   