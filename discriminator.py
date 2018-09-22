import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init

class Generator:

    def __init__(self,mini_batch_size : int ,side_pixels: int) -> None:

        pixels_per_image : int = side_pixels*side_pixels

        nodes_input_layer : int = 100

        self.G_W1 = tf.Variable(xavier_init([nodes_input_layer, mini_batch_size]) ) # , name="G_W1"
        self.G_b1 = tf.Variable(tf.zeros(shape=[mini_batch_size]) ) # , name="G_b1"

        self.G_W2 = tf.Variable(xavier_init([mini_batch_size, pixels_per_image ]) ) # , name="G_W2"
        self.G_b2 = tf.Variable(tf.zeros(shape=[784]) ) # , name="G_W2"
