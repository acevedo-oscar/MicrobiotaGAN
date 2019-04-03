import sys
sys.path.append("src/") 


import os
import random
import numpy as np
import sklearn.datasets
import tensorflow as tf
#import tensorflow_utils as tf_utils
import tensorflow_utils as tf_utils

import utils as utils
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

###


print(">> How many columns does the output data has?")
v1 = int(input())


print(">> How many GAN samples do you want?")
v2 = int(input())


print(">> Where is the saved model?")
filepath = input()

print(">> Type a name for the precdicted csv?")
v3 = input()
output_name ="/"+ v3+".csv"

##

DIM = 512  # model dimensionality
GEN_DIM = v1  # output dimension of the generator
DIS_DIM = 1  # outptu dimension fo the discriminator
FIXED_GENERATOR = False  # wheter to hold the generator fixed at ral data plus Gaussian noise, as in the plots in the paper
LAMBDA = .1  # smaller lambda makes things faster for toy tasks, but isn't necessary if you increase CRITIC_ITERS enough
BATCH_SIZE = 256   # batch size
ITERS = 100000 # how many generator iterations to train for
FREQ = 250  # sample frequency
 
# > ===== < 
sample_iter = math.ceil(v2/BATCH_SIZE) + 1
# > ===== <

img_folder = os.path.join('img','_' + str(FIXED_GENERATOR))


CRITIC_ITERS = 5  # homw many critic iteractions per generator iteration

if not os.path.isdir(img_folder):
    os.makedirs(img_folder)

utils.print_model_setting(locals().copy())

def Generator(n_samples,  name='gen'):
 
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
fake_data = Generator(BATCH_SIZE)

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

# Restoring session

session_saver = tf.train.Saver()

pre_trained  = 0

#Here
restore_folder = 'model/'+filepath + '/'
print("\n <== "+restore_folder+"===> \n")


iter_ = 1

# Inference
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Restore Model
    saver = tf.train.import_meta_graph(restore_folder+'my_gan.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(restore_folder))

    # End Restoring Model
    
    for k in range(sample_iter):
        print("Estamos en la iteracion "+str(k))
        fake_samples = sess.run(fake_data)

        print("Samples have been generated")
        print(fake_samples.shape)
        df = pd.DataFrame(fake_samples)
        with open('data/'+output_name, 'a') as f:
            df.to_csv(f, header=False, index=False)


    utils.flush(img_folder)
    # generate_image(sess, batch_data, iter_)
