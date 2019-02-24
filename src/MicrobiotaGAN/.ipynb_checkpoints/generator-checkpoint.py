import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init
from MicrobiotaGAN.batch_norm_wrapper import *


class Generator:

    def __init__(self, noise_dim : int, n_species: int) -> None:
         
        nodes_input_layer: int = 128
        self.epsilon = 1e-3

        self.G_W1 = tf.Variable(xavier_init([noise_dim, nodes_input_layer]), name="G_W1")
        self.G_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]), name="G_b1")

        self.L1_scale1 = tf.Variable(tf.ones([nodes_input_layer]), name="L1_scale1")
        self.L1_beta1 = tf.Variable(tf.zeros([nodes_input_layer]), name="L1_beta1")

        self.G_W2 = tf.Variable(xavier_init([nodes_input_layer, n_species]), name="G_W2")
        self.G_b2 = tf.Variable(tf.zeros(shape=[n_species]), name="G_b2")

    def train_draw_samples(self, noise, decay=tf.constant(0.999, dtype=tf.float32)):

        input_layer = tf.matmul(noise, self.G_W1) + self.G_b1

        normalized_input_layer = batch_norm_wrapper(input_layer, is_training=True)

        # ReLu :normalized_input_layer
        g_h1 = tf.nn.relu(normalized_input_layer)

        g_log_prob = tf.matmul(g_h1, self.G_W2) + self.G_b2

        normalized_g_log_prob = batch_norm_wrapper(g_log_prob, is_training=True)

        g_prob = tf.nn.sigmoid(normalized_g_log_prob)

        return g_prob

    def inference_draw_samples(self, noise):
        input_layer = tf.matmul(noise, self.G_W1) + self.G_b1

        normalized_input_layer = batch_norm_wrapper(input_layer, is_training=False)

        # ReLu :normalized_input_layer
        g_h1 = tf.nn.relu(normalized_input_layer)

        g_log_prob = tf.matmul(g_h1, self.G_W2) + self.G_b2

        normalized_g_log_prob = batch_norm_wrapper(g_log_prob, is_training=False)

        g_prob = tf.nn.sigmoid(normalized_g_log_prob)

        return g_prob

    def optimize_step(self, generator_cost) -> None:
        generator_parameters = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # tf.train.AdamOptimizer().minimize(generator_cost, var_list=generator_parameters)
        return tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(generator_cost, var_list=generator_parameters)
