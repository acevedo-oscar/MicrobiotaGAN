
#%%
"""
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

# %autoreload makes Jupyter to reload modules before executing the cell
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

"""
import copy
import os
working_dir= os.getcwd()


#%%

# os.chdir("../src")
print("We are at: "+os.getcwd()+"\n")


import sys
sys.path.append("../src") 

from MicrobiotaGAN.generator import Generator
from MicrobiotaGAN.discriminator import Discriminator
from MicrobiotaGAN.cost import wasserstein_generator_cost
from MicrobiotaGAN.cost import wasserstein_discriminator_cost
from MicrobiotaGAN.input_noise_sample import input_noise_sample
from MicrobiotaGAN.dataset_manager import DataSetManager 
from MicrobiotaGAN.glv_loss import GLV_Model
from MicrobiotaGAN.utilities import *
from MicrobiotaGAN.computational_graphs import *

from train_gan import train_gan

os.chdir(working_dir)

from timeit import default_timer as timer

#%%

import tensorflow as tf 

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns 
sns.set()


#%%
from snippets import *

#%% [markdown]
# ## Gaussian Distribution Data

#%%
def plot_normal(data, mu, sigma):
    count, bins, ignored = plt.hist(data, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),  linewidth=2, color='r')


#%%
mu, sigma = 0, 0.1 # mean and standard deviation
n_train_samples = 60000
numero_especies = 100

train_data = np.random.normal(mu, sigma, (n_train_samples,numero_especies))
print("Shape")
print(train_data.shape)
print("One samples mean")
print(np.mean(train_data[0,:]))

#%%
# Ejemplo de datos generados 
current = train_data[120,:]
print(current.shape)
plot_normal(current, mu, sigma)


#%%
microbiota_train_set = DataSetManager(train_data, norm=False)

#%% [markdown]
# ## Training Loop

#%%

n_species : int = numero_especies
mini_batch_size : int =32

noise_dim : int = 10
noise_sample = tf.placeholder(tf.float32, shape=[None, noise_dim])


#%%
noise_dim: int = 10
noise_sample = tf.placeholder(tf.float32, shape=[None, noise_dim])

print("==> Defining the Graph <==")

# Computation Graph Definition
my_generator = Generator(noise_dim, n_species)
my_discriminator = Discriminator(n_species)

train_real_sample, train_noise_sample, G_cost_train, G_train_step, D_cost_train, D_train_step, clip_D, D_logit_real_train, D_logit_fake_train, train_graph_saver = train_graph(
    my_discriminator, my_generator, n_species, noise_dim)

inference_real_sample, inference_noise_sample, G_cost_inference, D_cost_inference, clip_D, test_graph_saver = inference_graph(
    my_discriminator, my_generator, n_species, noise_dim)


#%%
graph_train_operations = [train_real_sample, train_noise_sample ,D_cost_train, clip_D, D_train_step, G_train_step, G_cost_train, train_graph_saver]

            
print("Insert desired epochs")

desired_epochs = int(input())
print("Desired Epochs: "+str(desired_epochs))

 
print("\tRestoring record  ")
recovered_status = 0


try:
    


    with open('model/loss_epochs.pkl', 'rb') as f:
        recovered = pickle.load(f)
        gen_cost_record, dis_cost_record, epochs_completed = recovered

    recovered_status = 1

except ValueError:
    print("\tSaved model wasn't found")

except OSError:
    print("\tSaved model wasn't found")


# Training Loop
if recovered_status == 1:
    d_train_cost = dis_cost_record
    g_train_cost = gen_cost_record

    microbiota_train_set.epochs_completed = epochs_completed
else: 
    d_train_cost = []
    g_train_cost = []

epoch_record = []

glv_std_error_record = []
glv_cost_record = []

total_epochs = desired_epochs

start = timer()

dis_cost, gen_cost = train_gan(microbiota_train_set, graph_train_operations, mini_batch_size, total_epochs , my_discriminator, my_generator, recovered_status)

print("Debug")
print(dis_cost)

d_train_cost += dis_cost
g_train_cost += gen_cost

"""
for e in range(len(dis_cost)):   
    d_train_cost.append(dis_cost[e])

for e in range(len(gen_cost)):    
    g_train_cost.append(gen_cost[e])
"""


epoch_record.append( microbiota_train_set.epochs_completed)


# print("Training epoch completed <"+str(k)+">"+" out of <"+str(total_epochs//5)+">")
print("Generator Loss: "+str(gen_cost[-1]))
print("Discriminator Loss: "+str(dis_cost[-1]))


end = timer()
print("==> Acumulated time "+str(end - start) + " s <==")

## End of Trainig Loop

#Exporting loss history

things_to_export = [g_train_cost, d_train_cost, total_epochs]

with open('model/loss_epochs.pkl', 'wb') as f:
  pickle.dump(things_to_export, f)


"""
#%%
muestra_gan = draw_gan_samples(my_generator, number_of_samples_to_draw=1)
muestra_gan = muestra_gan.T
print(muestra_gan.shape)
print(muestra_gan[0:5,0])


"""