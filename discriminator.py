import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init

class Discriminator:

    def __init__(self,mini_batch_size : int ,n_species: int) -> None: 

        nodes_input_layer : int = 128

        self.D_W1 = tf.Variable(xavier_init([n_species, nodes_input_layer]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]))

        self.D_W2 = tf.Variable(xavier_init([nodes_input_layer, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

    def get_probability_and_logit(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return [D_prob, D_logit]

    def optimize_step(self, discriminator_cost) -> None:
        discriminator_patameres = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        # tf.train.AdamOptimizer().minimize(discriminator_cost, var_list=discriminator_patameres)
        return  tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize( -1*discriminator_cost, var_list=discriminator_patameres)


    def clip_parameters(self, fixed_value:float):
        #Try a small value  like 0.01
        discriminator_patameres = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]
        clip_D = [p.assign(tf.clip_by_value(p, -fixed_value, fixed_value)) for p in discriminator_patameres]
        return clip_D
