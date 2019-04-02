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

from telegrad.dl_bot import DLBot

telegram_token = "753043252:AAG2wjtBKV9nlcv9VEjLDyoShzkTEjTKFzA"  # replace TOKEN with your bot's token

# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = 780738092 # replace None with your telegram user id (integer):


print("What's the name of the CSV to train?\n")
archivero = input()
 

DIM = 512  # model dimensionality
GEN_DIM = 100  # output dimension of the generator
DIS_DIM = 1  # outptu dimension fo the discriminator
FIXED_GENERATOR = False  # wheter to hold the generator fixed at ral data plus Gaussian noise, as in the plots in the paper
LAMBDA = .1  # smaller lambda makes things faster for toy tasks, but isn't necessary if you increase CRITIC_ITERS enough
BATCH_SIZE = 256   # batch size
ITERS = 100000 # how many generator iterations to train for
FREQ = 250  # sample frequency
 
img_folder = os.path.join('img','_' + str(FIXED_GENERATOR))


CRITIC_ITERS = 5  # homw many critic iteractions per generator iteration


def Generator_Softmax(n_samples,  name='gen'):

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
fake_data = Generator_Softmax(BATCH_SIZE)

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

# Dataset Loading

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

pre_trained  = False

# Create a DLBot instance
bot = DLBot(token=telegram_token, user_id=telegram_user_id)
# Activate the bot
bot.activate_bot()

print("\nTelegram bot has been activated ")

# Train loop
with tf.Session() as sess:
    
    if pre_trained: # false by default:
        sess.run(tf.global_variables_initializer())
    else:
        session_saver.restore(sess, 'model/my_gan.ckpt')
        
    
    
    
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

            bot.loss_hist.append(disc_cost_)

            
            fake_samples = sess.run(fake_data, feed_dict={real_data: batch_data})
            
            #fake_samples = fake_samples[0,:]
 
            print("\n==> Sum-Simplex condition: " +str(np.sum(fake_samples, axis=1)))    

            message = "\nEpochs ["+str(my_ds.epochs_completed)+"] Iter: "+str(iter_)+" , % "+str(100* iter_/ITERS) 
            print(message)
            bot.set_status(message)
            # Send update message
            if bot.verbose:
                bot.send_message(message)                
 
            # test_res = stats.ttest_ind(fake_samples,train_data[0,:])        
            # print("\n==> t-Test: " +str(test_res))
            print("\n")

            utils.flush(img_folder)
            # generate_image(sess, batch_data, iter_)
            
        if (np.mod(iter_, FREQ) == 0) or (iter_+1 == ITERS):

            session_saver.save(sess, 'model/my_gan.ckpt')

        utils.tick()  
