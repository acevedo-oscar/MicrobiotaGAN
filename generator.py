import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init


class Generator:`

    def __init__(self, noise_dim : int, n_species: int) -> None:
         
        nodes_input_layer : int = 128

        self.G_W1 = tf.Variable(xavier_init([noise_dim, nodes_input_layer]) , name="G_W1")
        self.G_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]), name="G_b1" )

        self.G_W2 = tf.Variable(xavier_init([nodes_input_layer, n_species ]) , name="G_W2")
        self.G_b2 = tf.Variable(tf.zeros(shape=[n_species]) , name="G_b2")

    def draw_samples(self, noise) :

        g_h1 = tf.nn.relu(tf.matmul(noise, self.G_W1) + self.G_b1)

        g_log_prob = tf.matmul(g_h1, self.G_W2) + self.G_b2
        g_prob = tf.nn.sigmoid(g_log_prob)

        return g_prob

    def optimize_step(self, generator_cost) -> None:
        generator_parameters = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # tf.train.AdamOptimizer().minimize(generator_cost, var_list=generator_parameters)
        return tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(generator_cost, var_list=generator_parameters)
