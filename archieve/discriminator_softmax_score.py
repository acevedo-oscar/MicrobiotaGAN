import sys
sys.path.append("src/") 

import tensorflow as tf
#import tensorflow_utils as tf_utils

from src import tensorflow_utils as tf_utils
from src import utils as utils
import math



# print(">> How many columns does the output data has?")
v1 = 100

# print(">> How many GAN samples do you want?")
v2 =1


 

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


def discriminator_score(data,model_filepath):

        """
        print(">> Where is the saved model?")
        model_filepath = input()

        #Here
        restore_folder = 'model/'+filepath + '/'
        print("\n <== "+restore_folder+"===> \n")
        """
###

        # Inference
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # Restore Model
                saver = tf.train.import_meta_graph(model_filepath+'my_gan.ckpt.meta')
                saver.restore(sess, tf.train.latest_checkpoint(model_filepath))

                # End Restoring Model


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



                disc_input = tf.placeholder(tf.float32, shape=[None, GEN_DIM])

                """
                disc_real = Discriminator(real_data, is_reuse=False)
                disc_fake = Discriminator(fake_data)
                """

                disc_score = Discriminator(disc_input)
##
                score = sess.run(disc_score, feed_dict={disc_input: data})

        return score

