import sys
sys.path.append("src/") 


import os
import random
import numpy as np
import sklearn.datasets
import tensorflow as tf
import tensorflow_utils as tf_utils
import utils as utils

import pandas as pd
from scipy import stats

from typing import List

from dataset_manager import DataSetManager

from telegrad.dl_bot import DLBot

telegram_token = "753043252:AAG2wjtBKV9nlcv9VEjLDyoShzkTEjTKFzA"  # replace TOKEN with your bot's token

# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = 780738092 # replace None with your telegram user id (integer):

"""
----------------8<-------------[ cut here ]------------------

------------------------------------------------
"""
def train_gan(train_set, indices: List, samples_per_N:int, repetition_n:int, batch_size: int =256):
    assert train_set.shape[0] > len(indices)

    print(train_set.shape)
    print(len(indices))

    my_ds = DataSetManager(train_set[indices], norm=False)


    # print("Set number of iterations to train\n")
    v5 = 1000


    # print("Use pretrained model? (0 means No, some number different to 0 means yes)\n")
    decision_number = 0 #int( input() )

    # print("Type a name to save the model with?\n")
    model_tag = str(round(samples_per_N)) +'_'+ str(repetition_n)
    

    storing_path = 'data/'+ model_tag + '_data/'
    model_path = storing_path+ model_tag + '.ckpt'

    dirName = storing_path
    
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    # ===> Auxiliar functions <=== 
    """
    ----------------8<-------------[ cut here ]------------------

    ------------------------------------------------
    """
    def save_history(files_prefix, gen_loss_record,disc_loss_record, epoch_record,my_ds,iter_, epochs, global_iters ):
        # Save losses per epoch

        df = pd.DataFrame(np.array(gen_loss_record))
        with open(files_prefix+'_gen_loss.csv', 'w+') as f:
            df.to_csv(f, header=False, index=False)

        df = pd.DataFrame(np.array(disc_loss_record))
        with open(files_prefix+'_disc_loss.csv', 'w+') as f:
            df.to_csv(f, header=False, index=False)

        df = pd.DataFrame(np.array(epoch_record))
        with open(files_prefix+'_epoch_record.csv', 'w+') as f:
            df.to_csv(f, header=False, index=False)
        # Save current iter and epochs

        df = pd.DataFrame(np.array( [epochs + my_ds.epochs_completed, global_iters + iter_] ) ) 

        with open(files_prefix+'_training.csv', 'w+') as f:
            df.to_csv(f, header=False, index=False)

    def send_bot_message(bot,my_ds, iter_, ITERS ):

        message = "\nEpochs ["+str(my_ds.epochs_completed)+"] Iter: "+str(iter_)+" , % "+str(100* iter_/ITERS) 
        print(message)
        bot.set_status(message)
        # Send update message
        if bot.verbose:
            bot.send_message(message)                

        print("\n")

    def save_gen_samples(gen_op, disc_op, sess,path,  k, n = 4):
        """
        k: is the number of epochs used to trained the generator
        n: is the number of batches to draw samples
        """

        suffix = '_gen_samples_'+str(k)+'_epochs_'+'.csv'

        for k in range(n):

            samples = sess.run(gen_op)
            df = pd.DataFrame(np.array(samples))
            with open(path+suffix, 'a') as f:
                df.to_csv(f, header=False, index=False)

            # Score the samples using the critic
            scores = sess.run(disc_op)
            df = pd.DataFrame(np.array(scores))
            with open(path+'scores_'+suffix, 'a') as f:
                df.to_csv(f, header=False, index=False)

    # ===> Model Parameters <=== 
    """
    ----------------8<-------------[ cut here ]------------------

    ------------------------------------------------
    """

    DIM = 512  # model dimensionality
    GEN_DIM = 100  # output dimension of the generator
    DIS_DIM = 1  # outptu dimension fo the discriminator
    FIXED_GENERATOR = False  # wheter to hold the generator fixed at ral data plus Gaussian noise, as in the plots in the paper
    LAMBDA = .1  # smaller lambda makes things faster for toy tasks, but isn't necessary if you increase CRITIC_ITERS enough
    BATCH_SIZE = batch_size   # batch size
    ITERS = v5 #100000 # how many generator iterations to train for
    FREQ = 250  # sample frequency
    
    


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

    """
    ----------------8<-------------[ cut here ]------------------

    ------------------------------------------------
    """
    # ===> Model Parameters <=== 

    df = pd.DataFrame(np.array(indices))
    with open(storing_path+'training_indices.csv', 'a') as f:
        df.to_csv(f, header=False, index=False)

  
    session_saver = tf.train.Saver()

    # files_prefix = 'model/'+ model_tag 

    if decision_number == 0:
        pre_trained  = False

        gen_loss_record = []  # type: List[float]
        disc_loss_record = []  # type: List[float]
        epoch_record = []  # type: List[float]

        epochs = 0
        global_iters = 0

    else:
        pre_trained  = True
        temp = pd.read_csv(storing_path+'_training.csv',header=None  ).values
        
        epochs, global_iters = temp.flatten()

        my_ds.epochs_completed  = epochs

        gen_loss_record = (pd.read_csv(storing_path+'_gen_loss.csv',header=None  ).values).tolist()
        disc_loss_record = (pd.read_csv(storing_path+'_disc_loss.csv',header=None  ).values).tolist()
        epoch_record = (pd.read_csv(storing_path+'_epoch_record.csv',header=None  ).values).tolist()

    # Create a DLBot instance
    bot = DLBot(token=telegram_token, user_id=telegram_user_id)
    # Activate the bot
    bot.activate_bot()

    print("\nTelegram bot has been activated ")
    
    # Train loop
    with tf.Session() as sess:
        
        if pre_trained == False: # false by default:
            sess.run(tf.global_variables_initializer())
        if pre_trained == True:
            # tf.reset_default_graph() 
            session_saver.restore(sess,model_path)
        
        for iter_ in range(ITERS):
            batch_data, disc_cost_ = None, None
            
            previous_epoch =  my_ds.epochs_completed 

            # train critic
            for i_ in range(CRITIC_ITERS):
                batch_data =  my_ds.next_batch(BATCH_SIZE) # data_gen.__next__()
                disc_cost_, _ = sess.run([disc_cost, disc_train_op], feed_dict={real_data: batch_data})
                
            # train generator
            sess.run(gen_train_op)   

            gen_cost2 = sess.run(gen_cost)   

            current_epoch =  my_ds.epochs_completed 

            condition2 = current_epoch % 5 == 0
            if current_epoch > previous_epoch and condition2:
                disc_loss_record.append(disc_cost_)
                gen_loss_record.append(gen_cost2)
                epoch_record.append(my_ds.epochs_completed ) 
                # print("Diff "+str(current_epoch - previous_epoch))

            if (np.mod(iter_, FREQ) == 0) or (iter_+1 == ITERS):
                
                """
                print("===> Debugging")
                print(disc_loss_record)
                print(gen_loss_record)
                """

                bot.loss_hist.append(disc_cost_)

                fake_samples = sess.run(fake_data) # , feed_dict={real_data: batch_data}
                # print("\n==> Sum-Simplex condition: " +str(np.sum(fake_samples, axis=1))) 
                send_bot_message(bot,my_ds, iter_, ITERS )
    
                session_saver.save(sess, model_path)
                save_history(storing_path, gen_loss_record,disc_loss_record,epoch_record, my_ds,iter_, epochs, global_iters )

                k = my_ds.epochs_completed
                # save_gen_samples(fake_data, disc_fake ,sess, storing_path, k) # fake_data = Generator_Softmax(BATCH_SIZE)
                

            utils.tick()  #  _iter[0] += 1

        if iter_ == ITERS:
            session_saver.save(sess, model_path)
        
        # Create gan samples
        n_samples = len(indices)

        k_iter = n_samples//BATCH_SIZE +1

        gan_samples_path = storing_path+"gan_samples_" +model_tag+'.csv'

        for k in range(k_iter):
            fake_samples = sess.run(fake_data)

            df = pd.DataFrame(fake_samples)
            with open(gan_samples_path, 'a') as f:
                df.to_csv(f, header=False, index=False)

    # Clear variables valuies

    tf.reset_default_graph()

    save_history(storing_path, gen_loss_record,disc_loss_record,epoch_record, my_ds,iter_, epochs, global_iters )
    k = my_ds.epochs_completed

    bot.stop_bot()

    print("Training is done")

    # Duct tapping the size of gan sample set to avoid changing the TF Graph

    temp1 = pd.read_csv('data/experiment_march_21/train_set.csv', header=None).values
    temp1 = temp1[0:n_samples]
    df = pd.DataFrame(temp1)

    with open(gan_samples_path, 'w+') as f:
        df.to_csv(f, header=False, index=False)


    print("Training is done")
