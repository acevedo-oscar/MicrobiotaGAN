#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
working_dir= os.getcwd()

# %autoreload makes Jupyter to reload modules before executing the cell
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


os.chdir("../src")

from src.MicrobiotaGAN.generator import Generator
from src.MicrobiotaGAN.discriminator import Discriminator

from src.MicrobiotaGAN.cost import wasserstein_generator_cost
from src.MicrobiotaGAN.cost import wasserstein_discriminator_cost

from src.MicrobiotaGAN.input_noise_sample import input_noise_sample

from src.MicrobiotaGAN.dataset_manager import DataSetManager

from src.MicrobiotaGAN.glv_loss import GLV_Model

from src.MicrobiotaGAN.utilities import *

os.chdir(working_dir)


# In[3]:


import tensorflow as tf 

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# In[4]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pickle

import seaborn as sns
 
sns.set()
from timeit import default_timer as timer


# In[ ]:




microbiota_table = pd.read_csv('../data/abundances.csv',header=None  ).values

 
    
param_path = '../data/image_dynamics_param.pickle' 

with open(param_path, 'rb') as pickleFile:
    microbiota_parameters = pickle.load(pickleFile)


# # Splitting data into training set and test set

# In[ ]:


# Load Data to the data mangaer

# Reduce data amount for quick training

microbiota_table = microbiota_table[0:2500]
#microbiota_table = np.multiply(microbiota_table, 0.1)

np.random.shuffle(microbiota_table)

training_percent = 0.9

lim = int(np.ceil(microbiota_table.shape[0]*training_percent))
train_set = microbiota_table[0:lim,:]
test_set = microbiota_table[lim:,:]
print("Test set" +str(test_set.shape))
print("Train set: "+str(train_set.shape))


microbiota_test_set =  DataSetManager(test_set, norm=True)
microbiota_train_set=  DataSetManager(train_set, norm=True)

m_sigma = microbiota_parameters['sigma']
m_rho = microbiota_parameters['rho']

print("\n")
print("Sigma: "+str(m_sigma))
print("Rho: "+str(m_rho))

m_A = microbiota_parameters['A']

m_r = microbiota_parameters['r']




# In[ ]:


#np.std(mbio_dict["max_vec"], ddof=1) 
#x = microbiota_test_set.next_batch(1)
#print(np.sum(x[0,:]))
#print(np.max(x[0,:]))


# In[ ]:


microbiota_parameters.keys()


# # Dataset exploration 

# In[ ]:



plt.axis(xmax=1, ymax=1)
plt.plot(m_sigma, m_rho,'ro')
 
plt.xlabel(r'Interspecies interaction [$\sigma$]') 
plt.ylabel(r'Interspecies probability [$\rho$]')

print("Distance from origin: "+ str(np.sqrt(m_rho**2 + m_sigma**2)))


# In[ ]:


plt.imshow(microbiota_parameters['A'], cmap="gray_r")
plt.title("A Matrix")


# In[ ]:


# Micorbiota abundance pseudo histogram
print(list(range(1,m_A.shape[0])))

sns.barplot(x = list(range(1,m_A.shape[0]+1)) ,y= np.sum(normalize_ds(microbiota_table), axis=0))
plt.title("Dataset Community Normalized Species Abundance")
plt.xlabel("Species")
plt.ylabel("Abundance")

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)


# # Computational Graph Definition

# In[ ]:


n_species : int = m_A.shape[0]
mini_batch_size : int =32

noise_dim : int = 10
noise_sample = tf.placeholder(tf.float32, shape=[None, noise_dim])


# Computation Graph Definition
my_generator = Generator(noise_dim, n_species)
my_discriminator = Discriminator(n_species)

real_sample = tf.placeholder(tf.float32, shape=[None, n_species])

 
D_real, norm_logit_t,  D_logit_real = my_discriminator.train_probability_and_logit(real_sample)


fake_gan_samples = my_generator.train_draw_samples(noise_sample)

# Train Graph
#generator_sample_train =  tf.placeholder(tf.float32, shape=[None, n_species], name="Fake_Input") #my_generator.train_draw_samples(noise_sample)
D_fake_train,  norm_logit_t,  D_logit_fake_train = my_discriminator.train_probability_and_logit(fake_gan_samples)

D_cost_train =  wasserstein_discriminator_cost(D_logit_real, D_logit_fake_train)
G_cost_train = wasserstein_generator_cost( D_logit_fake_train)

D_solver_train = my_discriminator.optimize_step(D_cost_train)
G_solver_train = my_generator.optimize_step(G_cost_train)

# Inference Graph
generator_sample_inference = my_generator.inference_draw_samples(noise_sample)
D_fake_inference, D_logit_fake_inference = my_discriminator.inference_probability_and_logit(generator_sample_inference)


D_cost_inference =  wasserstein_discriminator_cost(D_logit_real, D_logit_fake_inference)
G_cost_inference = wasserstein_generator_cost( D_logit_fake_inference)

D_solver_inference = my_discriminator.optimize_step(D_cost_inference)
G_solver_inference = my_generator.optimize_step(G_cost_inference)

# Others
clip_D = my_discriminator.clip_parameters(0.01)

# Draw train gan-fake samples
#draw_op = my_generator.inference_draw_samples(noise_sample)


# In[ ]:


# Initialize the network graph
# sess = tf.InteractiveSession() # tf.Session()
# sess.run(tf.global_variables_initializer())


# In[ ]:





# #Train Loop
# 

# In[ ]:

number_of_fig_per_plot :int = 16
# Training Loop
counter = 0
d_iter_ratio: int = 5

train_epochs = 20

iters_per_epoch = train_set.shape[0]//mini_batch_size

iteration_number = train_epochs*iters_per_epoch
# Since the discriminator is trained 5 times more, de divide the number of iterations
iteration_number = int(np.ceil(iteration_number/d_iter_ratio))

start = timer()

train_g_cost_record = []
train_d_cost_record = []
iter_record_g = []
iter_record_d = []

epoch_record_g = []
epoch = []

g_train_epoch_cost = []
d_train_epoch_cost =[]
 

initial_epochs = microbiota_train_set.epochs_completed

little_record = []

add_g_record = 0
D_current_cost = 0
G_current_cost = 0

#### Sess 1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    it = 0
    while microbiota_train_set.epochs_completed - initial_epochs < train_epochs:

        it += 1

        if microbiota_train_set.epochs_completed == 2:
            little_record.append(sess.run(my_generator.G_W1))

        # Train more the discriminator
        for k in range(d_iter_ratio):      

            previous_epoch = microbiota_train_set.epochs_completed
            real_sample_mini_batch = microbiota_train_set.next_batch(mini_batch_size)
            current_epoch = microbiota_train_set.epochs_completed

            dis_var_dict = {real_sample: real_sample_mini_batch,
                            noise_sample: input_noise_sample(mini_batch_size, noise_dim)}
            _, D_current_cost, _ = sess.run([D_solver_train, D_cost_train, clip_D], feed_dict=dis_var_dict)

            train_d_cost_record.append(D_current_cost)
            iter_record_d.append(it+1+k)

            if current_epoch > previous_epoch :
                d_train_epoch_cost.append(D_current_cost)
                add_g_record =1
                # g_train_epoch_cost.append(G_current_cost)

        # End For

        # Apply the optimization algorithm and update both network parameters
        gen_var_dict = {noise_sample: input_noise_sample(mini_batch_size, noise_dim)}
        _, G_current_cost = sess.run([G_solver_train, G_cost_train], feed_dict=gen_var_dict)    
        train_g_cost_record.append(G_current_cost)

        if add_g_record == 1:
            g_train_epoch_cost.append(G_current_cost)
            add_g_record = 0

        iter_record_g.append(it)

        if it % (iteration_number//10) == 0 or it == 0:
            end = timer()
            print("Progreso :"+str(100*it/(1.0*iteration_number))+" %")
            print(str(end - start)+" s")

            current_gen_weights = np.sum(sess.run(my_generator.G_W1))
            print("Current Gen Weights" +str(current_gen_weights))

            print("Epochs completed so far "+str(microbiota_train_set.epochs_completed))

            print('\t Iter: {}'.format(it))
            print('\t D loss: {:.4}'.format(D_current_cost))
            print('\t G_loss: {:.4}'.format(G_current_cost))

    saver = tf.train.Saver()
    saver.save(sess, '../results/trained_gan.ckpt')
    little_record.append(sess.run(my_generator.G_W1))


# In[ ]:


print(np.sum(little_record[0]))
print(np.sum(little_record[1]))


# In[ ]:


#plt.plot(iter_record_g, g_cost_record)


# In[ ]:


#plt.plot(iter_record_d, d_cost_record)


# In[ ]:





# # Test loop
# 

# In[ ]:




number_of_fig_per_plot :int = 16
#Training Loop
counter = 0

test_epochs = train_epochs

iters_per_epoch = test_set.shape[0]//mini_batch_size

test_iter =   test_epochs*iters_per_epoch



#test_iter =  (iteration_number*test_set.shape[0])//train_set.shape[0]

start = timer()

test_g_cost_record = []
test_d_cost_record = []
iter_test_record_g = []
iter_test_record_d = []

 
g_test_epoch_cost = []
d_test_epoch_cost =[]
 

epoch_record_g = []
epoch = []

d_iter_ratio: int = 5
 
print(iters_per_epoch)
print(test_iter)
print(test_set.shape[0])


with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())
 saver.restore(sess, '/content/trained_gan.ckpt')

 for it in range(test_iter):

     # Train more the discrimantor    
     for k in range(d_iter_ratio):      

         #f k%5 == 0 and k != 0:
             #pass

         real_sample_mini_batch = microbiota_test_set.next_batch(mini_batch_size)
         dis_var_dict = {real_sample: real_sample_mini_batch, noise_sample: input_noise_sample(mini_batch_size, noise_dim)}
         D_current_test_cost, _ = sess.run([ D_cost_inference, clip_D], feed_dict=dis_var_dict)
         test_d_cost_record.append(D_current_test_cost)
         iter_test_record_d.append(it+1+k)
         





     # End For



     # Apply the optimization algorithm and update both newtwork parameters
     gen_var_dict = {noise_sample: input_noise_sample(mini_batch_size, noise_dim)}
     temp_g_cost = sess.run([ G_cost_inference], feed_dict=gen_var_dict)
     #Since is the single fetch
     G_current_test_cost = temp_g_cost[0]

     test_g_cost_record.append(G_current_test_cost)

     iter_test_record_g.append(it)

     if it%iters_per_epoch == 0 and (k !=0):
         d_test_epoch_cost.append(D_current_test_cost)
         g_test_epoch_cost.append(G_current_test_cost)

     if it % (test_iter//10) == 0:
         end = timer()
         print("Progreso :"+str(100*it/(1.0*test_iter))+" %")
         print(str(end - start)+" s")

         print("epochs completed "+str(microbiota_test_set.epochs_completed))
         print(it/(iters_per_epoch/5))


         print('\t Iter: {}'.format(it))
         print('\t D loss: {:.4}'.format(D_current_test_cost))
         print('\t G_loss: {:.4}'.format(G_current_test_cost))


# # Generator Loss Graph

# In[ ]:


iter_line = list(range(len(train_g_cost_record)))
test_iter_line = list(range(len(test_g_cost_record)))

epochs_list = list(range(len(g_test_epoch_cost)))


#plt.plot(iter_line, train_g_cost_record )
#plt.plot(test_iter_line, test_g_cost_record)

plt.plot(epochs_list, g_train_epoch_cost)

plt.plot(epochs_list, g_test_epoch_cost)

plt.legend(['train set', 'test set'], loc='upper right')
plt.title("Generator Loss")

plt.xlabel("epochs")
plt.ylabel("Loss")

#train_g_cost_record


# In[ ]:


plt.plot(epochs_list, d_train_epoch_cost)

plt.plot(epochs_list, d_test_epoch_cost)


# ## Closer look on the interval [0,100]

# In[ ]:


plt.title("Generator loss on test set")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(iter_line[0:100],test_g_cost_record[0:100])


# # Discriminator Loss Graph

# In[ ]:


iter_line_d = list(range(len(train_d_cost_record)))

#plt.plot(iter_line_d,train_d_cost_record )
#plt.plot(iter_line_d,test_d_cost_record)


plt.plot(epochs_list, d_train_epoch_cost)

plt.plot(epochs_list, d_test_epoch_cost)


plt.legend(['train set', 'test set'], loc='upper right')
plt.title("Discriminator Loss")

plt.xlabel("Iterations")
plt.ylabel("Loss")

#train_g_cost_record


# ## Closer look on the interval [0,100]

# In[ ]:


plt.title("Discriminator loss on test set")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(iter_line_d[0:100],test_d_cost_record[0:100])


# # Saving the model and othe file management

# In[ ]:





# In[ ]:



#Comprimimos todos los archivos que se regenraron al guardar el moelo y despues lo descargamos
get_ipython().system('zip  bio_gan.zip bio_gan.*')

files.download("bio_gan.zip")


# In[ ]:


get_ipython().system('ls')


# In[ ]:



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '/content/trained_gan.ckpt')
    
    w1 = sess.run(my_generator.G_W1)
    b1 = sess.run(my_generator.G_b1)

    w2= sess.run(my_generator.G_W2)
    b2 = sess.run(my_generator.G_b2)



gen_weights_and_bias = [w1, b1, w2, b2]

print(w1)


# In[ ]:


#import pickle


# In[ ]:


with open('gen_param.pkl', 'wb') as f:
  pickle.dump(gen_weights_and_bias, f)


# # Using the Generator to create some samples

# In[ ]:


# generator_sample_inference = my_generator.inference_draw_samples(noise_sample)

noise_sample = tf.placeholder(tf.float32, shape=[None, noise_dim])
sample_nara = my_generator.inference_draw_samples(noise_sample)

    
n_samples = 10
samples_table = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess, '/content/trained_gan.ckpt')

    for k in range(n_samples):
        
        magico = input_noise_sample(1, 10)
        print(np.sum(magico))

        
        f_sam = sess.run(sample_nara, feed_dict={noise_sample :magico})
        samples_table.append(f_sam[0] )
        done_per =(k/(1.0*n_samples))*100
        #print(str(round(done_per,2))+"% has samples have been created")
        #print(k)
        if k%(n_samples//10)==0:
            print(str(done_per)+"% has samples have been created")

samples_table = np.array(samples_table)


# In[ ]:


sumatorias = [ np.sum(samples_table[k,:]) for k in range(samples_table.shape[0])]

prome = [ np.mean(samples_table[k,:]) for k in range(samples_table.shape[0])]
print(sumatorias[0:5])
print(prome[0:5])
print(len(prome))


# In[ ]:


samples_table = np.array(samples_table) 
    
    
ts_min = []
for row in range(samples_table.shape[0]):
    ts_min.append(np.min(samples_table[row,:]))
    
ts_max = []
for row in range(samples_table.shape[0]):
    ts_max.append(np.max(samples_table[row,:]))
    
ts_max = np.array(ts_max)
ts_min = np.array(ts_min)
    
new_samples_table =  []
for row in range(samples_table.shape[0]):

    _ = samples_table[row,:]*(ts_max[row]-ts_min[row])+ts_min[row]
    new_samples_table.append(_)
      
new_samples_table = np.array(new_samples_table)

 

glv_gen_errors = []
for k in range(samples_table.shape[0]):
    error = np.sum( GLV_Model(samples_table[k,:], m_A, m_r ))
    #print(error)
    glv_gen_errors.append(error)
    
glv_gen_errors = np.array(glv_gen_errors)

 


# In[ ]:


print("GAN Samples: "+str(samples_table.shape[0]))
print("GAN mean GLV error : "+str(np.mean(glv_gen_errors)))
print("GAN GLV error std : "+str(np.std(glv_gen_errors)))


# In[ ]:


sub = DataSetManager(train_set)


# In[ ]:


sub_train_set = sub.next_batch(samples_table.shape[0]) #train_set[0:n_samples,:]

glv_train_errors = []
for k in range(sub_train_set.shape[0]):
    error = np.sum( GLV_Model(sub_train_set[k,:], m_A, m_r ))
    #print(error)
    glv_train_errors.append(error)
    
glv_train_errors = np.array(glv_train_errors)

glv_mean_e = np.round(np.mean(glv_train_errors),4)
glv_e_std = np.round(np.std(glv_train_errors),4)


print("Training Samples <Normalized>: "+str(samples_table.shape[0]))
print("Sub Train Set mean GLV error : "+str(glv_mean_e))
print("Sub Train Set error std : "+str(glv_e_std))


# In[ ]:


diff_e = np.mean(glv_gen_errors)  - glv_mean_e 
diff_std = np.std(glv_gen_errors) - glv_e_std

print("Samples: "+str(samples_table.shape[0]))

print("GAN-Train mean GLV error : "+str(diff_e))
print("GAN-Train GLV error std : "+str(diff_std))

print("% GAN-Train mean GLV error : "+str(100*diff_e/glv_mean_e))
print("% GAN-Train GLV error std : "+str(100*diff_std/glv_e_std))


# In[ ]:


np.sum(samples_table < 0)


# In[ ]:


import pandas as pd

df = pd.DataFrame(samples_table)
with open('unormed.csv', 'a') as f:
    df.to_csv(f, header=False, index=False)
    
    

df = pd.DataFrame(new_samples_table)
with open('normalized.csv', 'a') as f:
    df.to_csv(f, header=False, index=False)    
    
get_ipython().system('zip gan_samples.zip *.csv')

files.download("gan_samples.zip")


# ## Asessing the GLV error

# In[ ]:


d = {'GAN': glv_gen_errors, 'Training': glv_train_errors }
df = pd.DataFrame(data=d)

ax = sns.boxplot( data=df)

plt.title("GLV Error, "+str(train_epochs)+" epochs , "+str(n_samples)+" samples draw")


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

plt.savefig("glv_barplot.png")


# In[ ]:


# !rm *.png
files.download("glv_barplot.png")


# # Misc

# In[ ]:


sns.barplot(x = list(range(1,51)) ,y= np.sum(normalize_ds(samples_table), axis=0))
plt.title("GAN-Dataset Community Normalized Species Abundance")
plt.xlabel("Species")
plt.ylabel("Abundance")
plt.savefig("fake.svg")


# In[ ]:



rnd_index = np.random.randint(low=1, high=140000,size=100)
sub_ds = normalize_ds(microbiota_table)[rnd_index]

sns.barplot(x = list(range(1,29)) ,y= np.sum(sub_ds, axis=0))
plt.title("Dataset Community Normalized Species Abundance")
plt.xlabel("Species")
plt.ylabel("Abundance")
plt.savefig("real.svg")
 


# In[ ]:


files.download("real.svg")
files.download("fake.svg")


# In[ ]:


a_t = normalize_ds(samples_table)


# In[ ]:


show_rounded_array(a_t[8,:],2)


# In[ ]:


get_ipython().system('zip  pics.zip *.svg')
files.download("pics.zip")


# In[ ]:


get_ipython().system('head -n 20 MicrobiotaGAN/generator.py')


# In[ ]:


bool(1)


# In[ ]:


dir(my_generator)


# In[ ]:


def Diagonal_Matrix(input_vector):
     return np.multiply(np.identity(input_vector.shape[0]),input_vector)


# In[ ]:


s_0 = normalize_ds(microbiota_table)[0]

res = np.multiply(Diagonal_Matrix(s_0), np.matmul(m_A,s_0)+m_rho)

print(res[0,:])

plt.imshow(res, cmap="gray")


"""
print(m_A.shape)
print(s_0.shape)
print(s_0)
"""


# In[ ]:


get_ipython().system('head -n 25 MicrobiotaGAN/glv_loss.py')


# In[ ]:




