import tensorflow as tf

import numpy as np
import pandas as pd
import sys
sys.path.append("../../src/") 
import os

import tensorflow_utils as tf_utils
import utils as utils


def generate_samples(model_name):

    DIM = 512  # model dimensionality
    GEN_DIM = 100  # output dimension of the generator
    BATCH_SIZE = 256   # batch size

    def Generator_Softmax(n_samples,  name='gen'):

        with tf.variable_scope(name):
            noise = tf.random_normal([n_samples, GEN_DIM])
            output01 = tf_utils.linear(noise, 2*DIM, name='fc-1')
            output01 = tf_utils.relu(output01, name='relu-1')
            
            output02 = tf_utils.linear(output01, 2*DIM, name='fc-2')
            output02 = tf_utils.relu(output02, name='relu-2')
            
            output03 = tf_utils.linear(output02, 2*DIM, name='fc-3')
            output03 = tf_utils.relu(output03, name='relu-3')

            output04 = tf_utils.linear(output03, GEN_DIM, name='fc-4')

            # Reminder: a logit can be modeled as a linear function of the predictors
            output05 = tf.nn.softmax(output04, name = 'softmax-1')

            return output05

    fake_data = Generator_Softmax(BATCH_SIZE)

    session_saver = tf.train.Saver()

    with tf.Session() as sess:


        session_saver.restore(sess,model_name+'.ckpt')

        print("Generating...")
        for k in range(195):
            fake_samples = sess.run(fake_data)

            df = pd.DataFrame(fake_samples)

            with open( 'gan_samples_'+model_name+ '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)


# Crawl thorugh folders


old_path = os.getcwd()

cwd = os.getcwd()

directory_list=[x[0].replace(cwd,'').replace('/','') for x in os.walk(cwd,topdown=True)]
directory_list= directory_list[1:]
directory_list = sorted(directory_list)

#print(directory_list)
    
distance_record = []

id_sample_N = []
id_repetition = []

# print(len(directory_list))
# For loop goes here
for dir_k in range(len(directory_list)):

    if dir_k%5 ==0:
        percentage = str(np.round(100*dir_k /len(directory_list),2))+"%"
        print("Generating Samples "+percentage )

    numbers_list =(directory_list[dir_k].replace('_data','').replace('_',' ').replace('/',' ')).split(' ')

    os.chdir(directory_list[dir_k])
    ids = [int(numbers_list[k]) for k in range(len(numbers_list))]

    samples_file =numbers_list[0]+'_'+numbers_list[1] 

    generate_samples(samples_file)
    tf.reset_default_graph()

 
    
    os.chdir(cwd)
    
    ## 

os.chdir(old_path)

