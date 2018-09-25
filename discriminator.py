import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init

class Discriminator:

    def __init__(self,mini_batch_size : int ,side_pixels: int) -> None:

        pixels_per_image : int = side_pixels*side_pixels

        nodes_input_layer : int = 100

        self.D_W1 = tf.Variable(xavier_init([pixels_per_image, mini_batch_size]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[mini_batch_size]))

        self.D_W2 = tf.Variable(xavier_init([mini_batch_size, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit
