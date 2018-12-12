import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init
from MicrobiotaGAN.batch_norm_wrapper import *


class Discriminator:

    def __init__(self, n_species: int) -> None:

        nodes_input_layer: int = 128
        self.epsilon = 1e-3

        self.D_W1 = tf.Variable(xavier_init([n_species, nodes_input_layer]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]))

        self.L1_scale1 = tf.Variable(tf.ones([nodes_input_layer]))
        self.L1_beta1 = tf.Variable(tf.zeros([nodes_input_layer]))

        self.D_W2 = tf.Variable(xavier_init([nodes_input_layer, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

    def train_probability_and_logit(self, x, decay=0.999):
        input_layer = tf.matmul(x, self.D_W1) + self.D_b1

        normalized_input_layer = batch_norm_wrapper(input_layer, is_training=True)

        d_h1 = tf.nn.relu(normalized_input_layer)

        d_logit = tf.matmul(d_h1, self.D_W2) + self.D_b2

        return d_logit

    def inference_probability_and_logit(self, x):
        input_layer = tf.matmul(x, self.D_W1) + self.D_b1

        normalized_input_layer = batch_norm_wrapper(input_layer, is_training=False)

        d_h1 = tf.nn.relu(normalized_input_layer)

        d_logit = tf.matmul(d_h1, self.D_W2) + self.D_b2

        return d_logit

    def optimize_step(self, discriminator_cost) -> None:
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        # tf.train.AdamOptimizer().minimize(discriminator_cost, var_list=discriminator_parameters)
        # I DOES goes this way: -1*discriminator_cost
        return tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-1*discriminator_cost, var_list=discriminator_parameters)

    def clip_parameters(self, fixed_value:float):
        # Try a small value  like 0.01
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]
        clip_d = [p.assign(tf.clip_by_value(p, -fixed_value, fixed_value)) for p in discriminator_parameters]
        return clip_d
