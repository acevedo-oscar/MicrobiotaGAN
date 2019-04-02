import pandas as pd
import numpy as np

print("Loading Train func")
from train_dirichlet_interface import train_gan
print("Finished Loading Train func")

print(np.random.random_sample())

train_ds = pd.read_csv('data/experiment_march_21/train_set.csv', header=None) .values

ds_size = 1000

train_ds = train_ds[0:ds_size,:]

repetitions = 2
batch_size = 256

partiton_n = [300, 400, 500, 700]

for m in range(len(partiton_n)):

    for k in range(repetitions):

        # partiton_n = np.random.randint(batch_size,ds_size) #At least it has to fit one batch size
        index_list = np.random.randint(0,ds_size, size=partiton_n[m]).tolist()

        print("Calling training function")
        train_gan(train_ds,  index_list, partiton_n[m], k)
        print("====> Finished repetition "+str(k)+' of partition # '+str(m))

