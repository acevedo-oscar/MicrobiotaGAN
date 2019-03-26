import pandas as pd
import numpy as np

print("Loading Train func")
from train_dirichlet_interface import train_gan
print("Finished Loading Train func")

train_ds = pd.read_csv('data/experiment_march_21/train_set.csv', header=None).values
print(train_ds.shape)
ds_size = train_ds.shape[0]

# train_ds = train_ds[0:ds_size,:]

repetitions = 5
batch_size = 256

"""
partiton_n = [np.random.randint(batch_size,ds_size) for k in range(10) ] 

df = pd.DataFrame(np.array(partiton_n))

save_name = 'random_amounts.csv'
with open(save_name, 'a') as f:
    df.to_csv(f, header=False, index=False)
"""

partiton_n = [300, 400, 500, 700]

# partiton_n = pd.read_csv('random_amounts.csv', header=None).values.flatten()

print(partiton_n)
print(len(partiton_n))

for m in range(len(partiton_n)):

    for k in range(repetitions):

        # partiton_n = np.random.randint(batch_size,ds_size) #At least it has to fit one batch size
        index_list = np.random.randint(0,ds_size, size=partiton_n[m]).tolist()

        print("Calling training function")
        train_gan(train_ds,  index_list, partiton_n[m], k)
        print("====> Finished repetition "+str(k)+' of partition # '+str(m))
