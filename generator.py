import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init


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

        self.L2_scale2 = tf.Variable(tf.ones([n_species]), name="L2_scale2")
        self.L2_beta2 = tf.Variable(tf.zeros([n_species]), name="L2_beta2")

    def train_draw_samples(self, noise, decay=0.999):

        input_layer = tf.matmul(noise, self.G_W1) + self.G_b1

        pop_mean1 = tf.Variable(tf.zeros([input_layer.get_shape()[-1]]), trainable=False)
        pop_var1 = tf.Variable(tf.ones([input_layer.get_shape()[-1]]), trainable=False)

        batch_mean1, batch_var1 = tf.nn.moments(input_layer, [0])

        train_mean1 = tf.assign(pop_mean1, pop_mean1 * decay + batch_mean1 * (1 - decay))
        train_var1 = tf.assign(pop_var1, pop_var1 * decay + batch_var1 * (1 - decay))

        with tf.control_dependencies([train_mean1, train_var1]):
            normalized_input_layer = tf.nn.batch_normalization(input_layer,
                                                               batch_mean1,
                                                               batch_var1,
                                                               self.L1_scale1,
                                                               self.L1_beta1,
                                                               self.epsilon)
        g_h1 = tf.nn.relu(normalized_input_layer)

        g_log_prob = tf.matmul(g_h1, self.G_W2) + self.G_b2

        pop_mean2 = tf.Variable(tf.zeros([g_log_prob.get_shape()[-1]]), trainable=False)
        pop_var2 = tf.Variable(tf.ones([g_log_prob.get_shape()[-1]]), trainable=False)

        batch_mean2, batch_var2 = tf.nn.moments(g_log_prob, [0])

        train_mean2 = tf.assign(pop_mean2, pop_mean2 * decay + batch_mean2 * (1 - decay))
        train_var2 = tf.assign(pop_var2, pop_var2 * decay + batch_var2 * (1 - decay))

        with tf.control_dependencies([train_mean2, train_var2]):
            normalized_g_log_prob = tf.nn.batch_normalization(g_log_prob,
                                                           batch_mean2,
                                                           batch_var2,
                                                           self.L2_scale2,
                                                           self.L2_beta2,
                                                           self.epsilon)

        g_prob = tf.nn.sigmoid(normalized_g_log_prob)

        return g_prob

    def inference_draw_samples(self, noise):
        input_layer = tf.matmul(noise, self.G_W1) + self.G_b1

        pop_mean1 = tf.Variable(tf.zeros([input_layer.get_shape()[-1]]), trainable=False)
        pop_var1 = tf.Variable(tf.ones([input_layer.get_shape()[-1]]), trainable=False)

        normalized_input_layer = tf.nn.batch_normalization(input_layer,
                                                           pop_mean1,
                                                           pop_var1,
                                                           self.L1_scale1,
                                                           self.L1_beta1,
                                                           self.epsilon)
        g_h1 = tf.nn.relu(normalized_input_layer)

        g_log_prob = tf.matmul(g_h1, self.G_W2) + self.G_b2

        pop_mean2 = tf.Variable(tf.zeros([g_log_prob.get_shape()[-1]]), trainable=False)
        pop_var2 = tf.Variable(tf.ones([g_log_prob.get_shape()[-1]]), trainable=False)

        normalized_g_log_prob = tf.nn.batch_normalization(g_log_prob,
                                                          pop_mean2,
                                                          pop_var2,
                                                          self.L2_scale2,
                                                          self.L2_beta2,
                                                          self.epsilon)

        g_prob = tf.nn.sigmoid(normalized_g_log_prob)

        return g_prob

    def optimize_step(self, generator_cost) -> None:
        generator_parameters = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # tf.train.AdamOptimizer().minimize(generator_cost, var_list=generator_parameters)
        return tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(generator_cost, var_list=generator_parameters)
