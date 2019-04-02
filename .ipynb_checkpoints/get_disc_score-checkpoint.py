


import tensorflow as tf
import numpy as np


# In[2]:


get_ipython().system('pwd')
# model_path = "model/dir1/my_gan.ckpt.meta"

import sys
sys.path.append("src/") 

#import tensorflow_utils as tf_utils

from src import tensorflow_utils as tf_utils
from src import utils as utils
 


# In[9]:



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


DIM = 512  # model dimensionality
GEN_DIM = 100  # output dimension of the generator
DIS_DIM = 1  # outptu dimension fo the discriminator

# model_path =  "model/dir1/my_gan.ckpt"

def get_disc_score(input_data,restore_path ):
    
    
    real_data = tf.placeholder(tf.float32, shape=[None, input_data.shape[1]])

    disc_real = Discriminator(real_data, is_reuse=False)
    saver = tf.train.Saver()


    with tf.Session() as sess:
        saver.restore(sess, restore_path)



        value = sess.run(disc_real, feed_dict={real_data: input_data})
        
    tf.reset_default_graph() 
    
    return value
