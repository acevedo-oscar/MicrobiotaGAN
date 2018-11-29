import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init


class Discriminator:

    def __init__(self,mini_batch_size : int ,n_species: int) -> None: 

        nodes_input_layer : int = 128

        self.D_W1 = tf.Variable(xavier_init([n_species, nodes_input_layer]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]))

        self.D_W2 = tf.Variable(xavier_init([nodes_input_layer, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

    def get_probability_and_logit(self, x, train=False):
        input_layer = tf.matmul(x, self.D_W1) + self.D_b1
        normalized_input_layer = tf.layers.batch_normalization(input_layer, training=train)
        d_h1 = tf.nn.relu(normalized_input_layer)

        d_logit = tf.matmul(d_h1, self.D_W2) + self.D_b2
        normalized_d_logit = tf.layers.batch_normalization(d_logit, training=train)
        d_prob = tf.nn.sigmoid(normalized_d_logit)

        return [d_prob, d_logit]

    def optimize_step(self, discriminator_cost) -> None:
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        # tf.train.AdamOptimizer().minimize(discriminator_cost, var_list=discriminator_parameters)
        return tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-1*discriminator_cost, var_list=discriminator_parameters)

    def clip_parameters(self, fixed_value:float):
        # Try a small value  like 0.01
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]
        clip_d = [p.assign(tf.clip_by_value(p, -fixed_value, fixed_value)) for p in discriminator_parameters]
        return clip_d
