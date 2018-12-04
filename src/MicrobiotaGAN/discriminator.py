import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init


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

        self.L2_scale2 = tf.Variable(tf.ones([n_species]))
        self.L2_beta2 = tf.Variable(tf.zeros([n_species]))

        self.pop_mean1 = tf.Variable(tf.zeros([nodes_input_layer]), trainable=False)
        self.pop_var1 = tf.Variable(tf.ones([nodes_input_layer]), trainable=False)

        self.pop_mean2 = tf.Variable(tf.zeros([1]), trainable=False)
        self.pop_var2 = tf.Variable(tf.ones([1]), trainable=False)

    def train_probability_and_logit(self, x, decay=0.999):
        input_layer = tf.matmul(x, self.D_W1) + self.D_b1

        # pop_mean1 = tf.Variable(tf.zeros([input_layer.get_shape()[-1]]), trainable=False)
        # pop_var1 = tf.Variable(tf.ones([input_layer.get_shape()[-1]]), trainable=False)

        batch_mean1, batch_var1 = tf.nn.moments(input_layer, [0])

        train_mean1 = tf.assign(self.pop_mean1, self.pop_mean1 * decay + batch_mean1 * (1 - decay))
        train_var1 = tf.assign(self.pop_var1, self.pop_var1 * decay + batch_var1 * (1 - decay))

        with tf.control_dependencies([train_mean1, train_var1]):
            normalized_input_layer = tf.nn.batch_normalization(input_layer,
                                                               batch_mean1,
                                                               batch_var1,
                                                               self.L1_scale1,
                                                               self.L1_beta1,
                                                               self.epsilon)
        # Relu
        d_h1 = tf.nn.relu(normalized_input_layer)

        d_logit = tf.matmul(d_h1, self.D_W2) + self.D_b2

        # pop_mean2 = tf.Variable(tf.zeros([d_logit.get_shape()[-1]]), trainable=False)
        # pop_var2 = tf.Variable(tf.ones([d_logit.get_shape()[-1]]), trainable=False)

        batch_mean2, batch_var2 = tf.nn.moments(d_logit, [0])

        train_mean2 = tf.assign(self.pop_mean2, self.pop_mean2 * decay + batch_mean2 * (1 - decay))
        train_var2 = tf.assign(self.pop_var2, self.pop_var2 * decay + batch_var2 * (1 - decay))

        with tf.control_dependencies([train_mean2, train_var2]):
            normalized_d_logit = tf.nn.batch_normalization(d_logit,
                                                           batch_mean2,
                                                           batch_var2,
                                                           self.L2_scale2,
                                                           self.L2_beta2,
                                                           self.epsilon)
        d_prob = tf.nn.sigmoid(normalized_d_logit)

        return [d_prob, normalized_d_logit, d_logit, input_layer]

    def inference_probability_and_logit(self, x):
        input_layer = tf.matmul(x, self.D_W1) + self.D_b1

        # pop_mean1 = tf.Variable(tf.zeros([input_layer.get_shape()[-1]]), trainable=False)
        # pop_var1 = tf.Variable(tf.ones([input_layer.get_shape()[-1]]), trainable=False)
        # batch_mean1, batch_var1 = tf.nn.moments(input_layer, [0])

        normalized_input_layer = tf.nn.batch_normalization(input_layer,
                                                           self.pop_mean1,
                                                           self.pop_var1,
                                                           self.L1_scale1,
                                                           self.L1_beta1,
                                                           self.epsilon)
        d_h1 = tf.nn.relu(normalized_input_layer)

        d_logit = tf.matmul(d_h1, self.D_W2) + self.D_b2

        # pop_mean2 = tf.Variable(tf.zeros([d_logit.get_shape()[-1]]), trainable=False)
        # pop_var2 = tf.Variable(tf.ones([d_logit.get_shape()[-1]]), trainable=False)
        # batch_mean2, batch_var2 = tf.nn.moments(d_logit, [0])
        normalized_d_logit = tf.nn.batch_normalization(d_logit,
                                                       self.pop_mean2,
                                                       self.pop_var2,
                                                       self.L2_scale2,
                                                       self.L2_beta2,
                                                       self.epsilon)
        d_prob = tf.nn.sigmoid(normalized_d_logit)

        return [d_prob, normalized_d_logit, d_logit]

    def optimize_step(self, discriminator_cost) -> None:
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        # tf.train.AdamOptimizer().minimize(discriminator_cost, var_list=discriminator_parameters)
        # I does goes this way: -1*discriminator_cost
        return tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-1*discriminator_cost, var_list=discriminator_parameters)

    def clip_parameters(self, fixed_value:float):
        # Try a small value  like 0.01
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]
        clip_d = [p.assign(tf.clip_by_value(p, -fixed_value, fixed_value)) for p in discriminator_parameters]
        return clip_d
