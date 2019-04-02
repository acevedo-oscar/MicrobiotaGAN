import sys
sys.path.append("src/") 


import os
import random
import numpy as np
import sklearn.datasets
import tensorflow as tf
import tensorflow_utils as tf_utils
import utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from dataset_manager import DataSetManager

print("What's the name of the CSV to train?\n")
archivero = input()
 

DIM = 512  # model dimensionality
GEN_DIM = 100  # output dimension of the generator
DIS_DIM = 1  # outptu dimension fo the discriminator
FIXED_GENERATOR = False  # wheter to hold the generator fixed at ral data plus Gaussian noise, as in the plots in the paper
LAMBDA = .1  # smaller lambda makes things faster for toy tasks, but isn't necessary if you increase CRITIC_ITERS enough
BATCH_SIZE = 256*4  # batch size
ITERS = 100000//4  # how many generator iterations to train for
FREQ = 250  # sample frequency
 
img_folder = os.path.join('img','_' + str(FIXED_GENERATOR))


CRITIC_ITERS = 5  # homw many critic iteractions per generator iteration

if not os.path.isdir(img_folder):
    os.makedirs(img_folder)

utils.print_model_setting(locals().copy())


# In[3]:


def Generator(n_samples, real_data_, name='gen'):
    if FIXED_GENERATOR:
        return real_data_ + (1. * tf.random_normal(tf.shape(real_data_)))
    else:
        with tf.variable_scope(name):
            noise = tf.random_normal([n_samples, GEN_DIM])
            output01 = tf_utils.linear(noise, DIM, name='fc-1')
            output01 = tf_utils.relu(output01, name='relu-1')
            
            output02 = tf_utils.linear(output01, DIM, name='fc-2')
            output02 = tf_utils.relu(output02, name='relu-2')
            
            output03 = tf_utils.linear(output02, DIM, name='fc-3')
            output03 = tf_utils.relu(output03, name='relu-3')
            
            output04 = tf_utils.linear(output03, GEN_DIM, name='fc-4')

            # Reminder: a logit can be modeled as a linear function of the predictors
            output05 = tf.nn.softmax(output04, name = 'softmax-1')

            
            return output05
        

def Discriminator(inputs, is_reuse=True, name='disc'):
    with tf.variable_scope(name, reuse=is_reuse):
        print('is_reuse: {}'.format(is_reuse))
        output01 = tf_utils.linear(inputs, DIM, name='fc-1')
        output01 = tf_utils.relu(output01, name='relu-1')

        output02 = tf_utils.linear(output01, DIM, name='fc-2')
        output02 = tf_utils.relu(output02, name='relu-2')

        output03 = tf_utils.linear(output02, DIM, name='fc-3')
        output03 = tf_utils.relu(output03, name='relu-3')

        output04 = tf_utils.linear(output03, DIS_DIM, name='fc-4')
        
        return output04
    
real_data = tf.placeholder(tf.float32, shape=[None, GEN_DIM])
fake_data = Generator(BATCH_SIZE, real_data)

disc_real = Discriminator(real_data, is_reuse=False)
disc_fake = Discriminator(fake_data)

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = - tf.reduce_mean(disc_fake)

# WGAN gradient penalty parameters

alpha = tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
interpolates = alpha*real_data + (1.-alpha) * fake_data
disc_interpolates = Discriminator(interpolates)
gradients = tf.gradients(disc_interpolates, [interpolates][0])
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes - 1)**2)

disc_cost += LAMBDA * gradient_penalty
    
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')


#WGAN Training operations
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_vars)

if len(gen_vars) > 0:
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_vars)
else:
    gen_train_op = tf.no_op()

def generate_image(sess, true_dist, idx):
    # generates and saves a plot of the true distribution, the generator, and the critic
    N_POINTS = 128
    RANGE = 2
    
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    
    if FIXED_GENERATOR is not True:
        samples = sess.run(fake_data, feed_dict={real_data: points})
    disc_map = sess.run(disc_real, feed_dict={real_data: points})
    
    plt.clf()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.colorbar()  # add color bar
    
    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    if FIXED_GENERATOR is not True:
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='*')
        
    plt.savefig(os.path.join(img_folder, str(idx).zfill(3) + '.jpg'))

 
# Dataset iterator
def inf_train_gen():
    if dataset == '8gaussians':
        scale = 2.
        centers = [(1.,0.), 
                   (-1.,0.), 
                   (0., 1.), 
                   (0.,-1.),
                   (1./np.sqrt(2), 1./np.sqrt(2)),
                   (1./np.sqrt(2), -1/np.sqrt(2)), 
                   (-1./np.sqrt(2), 1./np.sqrt(2)), 
                   (-1./np.sqrt(2), -1./np.sqrt(2))]
        
        centers = [(scale*x, scale*y) for x, y in centers]
        while True:
            batch_data = []
            for _ in range(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                batch_data.append(point)
                
            batch_data = np.array(batch_data, dtype=np.float32)
            batch_data /= 1.414  # std
            yield batch_data
            
    elif dataset == '25gaussians':
        batch_data = []
        for i_ in range(4000):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    batch_data.append(point)
                    
        batch_data = np.asarray(batch_data, dtype=np.float32)
        np.random.shuffle(batch_data)
        batch_data /= 2.828  # std
        
        while True:
            for i_ in range(int(len(batch_data)/BATCH_SIZE)):
                yield batch_data[i_*BATCH_SIZE:(i_+1)*BATCH_SIZE]
                
    elif dataset == 'swissroll':
        while True:
            batch_data = sklearn.datasets.make_swiss_roll(n_samples=BATCH_SIZE, noise=0.25)[0]
            batch_data = batch_data.astype(np.float32)[:, [0, 2]]
            batch_data /= 7.5  # stdev plus a little
            yield batch_data

# Dataets

mu, sigma = 0, 0.1 # mean and standard deviation
n_train_samples = 60000
numero_especies = 100

print("==> Loading CSV")
v1 = 'data/'+str(archivero)
print(v1)
train_data =  pd.read_csv(v1,header=None  ).values # np.random.normal(mu, sigma, (n_train_samples,numero_especies))
mean_vec = np.zeros(100) # vector de 100 ceros

print("Shape")
print(train_data.shape)
print("One samples mean")
print(np.mean(train_data[0,:]))

my_ds = DataSetManager(train_data, norm=False)


session_saver = tf.train.Saver()

pre_trained  = 0


# Train loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data_gen = inf_train_gen()
   
    
    for iter_ in range(ITERS):
        batch_data, disc_cost_ = None, None
        
        # train critic
        for i_ in range(CRITIC_ITERS):
            batch_data =  my_ds.next_batch(BATCH_SIZE) # data_gen.__next__()
            disc_cost_, _ = sess.run([disc_cost, disc_train_op], feed_dict={real_data: batch_data})
             
        # train generator
        sess.run(gen_train_op)
        
        # write logs and svae samples
        utils.plot('disc cost', disc_cost_)
        
        if (np.mod(iter_, FREQ) == 0) or (iter_+1 == ITERS):
            
            fake_samples = sess.run(fake_data, feed_dict={real_data: batch_data})
            
            fake_samples = fake_samples[0,:]
            print("\n==> FAKE SAMPLES: " +str(fake_samples))
            print("\n==> Mean SAMPLES: " +str(np.mean(fake_samples)))    
            print("\n==> STD SAMPLES: " +str(np.std(fake_samples)))
            # test_res = stats.ttest_ind(fake_samples,train_data[0,:])        
            # print("\n==> t-Test: " +str(test_res))
            print("\n")

            utils.flush(img_folder)
            # generate_image(sess, batch_data, iter_)
            
        if (np.mod(iter_, FREQ) == 0) or (iter_+1 == ITERS):

            session_saver.save(sess, 'model/my_gan.ckpt')

        utils.tick()  
