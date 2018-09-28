import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init

class Generator:

    def __init__(self,noise_dim : int ,side_pixels: int) -> None:

        pixels_per_image : int = side_pixels*side_pixels

        nodes_input_layer : int = 128

        self.G_W1 = tf.Variable(xavier_init([noise_dim, nodes_input_layer]) )
        self.G_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]) )

        self.G_W2 = tf.Variable(xavier_init([nodes_input_layer, pixels_per_image ]) )
        self.G_b2 = tf.Variable(tf.zeros(shape=[pixels_per_image]) )

    def draw_samples(self, noise) :

        G_h1 = tf.nn.relu(tf.matmul(noise, self.G_W1) + self.G_b1)


        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    def optimize_step(self, generator_cost) -> None:
        generator_patameres = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # tf.train.AdamOptimizer().minimize(generator_cost, var_list=generator_patameres)
        return  tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(generator_cost, var_list=generator_patameres)
