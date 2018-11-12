import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init

class Discriminator:

    def __init__(self,mini_batch_size : int ,n_species: int) -> None:

        nodes_input_layer : int = 128

        self.D_W1 = tf.Variable(xavier_init([n_species, nodes_input_layer]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]))

        self.D_W2 = tf.Variable(xavier_init([nodes_input_layer, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

    def get_probability_and_logit(self, x  ):

        #train_bool = True

        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        # tf.layers.batch_normalization(D_h1, training=train_bool)

        mean_D_h1, var_D_h1 = tf.nn.moments(D_h1, axes = [0])
        norm_D_h1 = tf.nn.batch_normalization(D_h1, mean=mean_D_h1,
                                             variance=var_D_h1, offset=0.0,
                                             scale = 1.0, variance_epsilon= 1e-8 )

        D_logit = tf.matmul(norm_D_h1, self.D_W2) + self.D_b2
        #norm_D_D_logit = tf.layers.batch_normalization(D_logit, training=train_bool)


        #mean_D_logit, var_D_logit = tf.nn.moments(D_logit, axes = [0])
        
        """
        norm_D_logit = tf.nn.batch_normalization(D_logit, mean=mean_D_logit,
                                             variance=var_D_logit, offset=0.0,
                                             scale = 1.0, variance_epsilon= 1e-8 )
        """

        D_prob = tf.nn.sigmoid(D_logit)

        return [D_prob, D_logit]

    def optimize_step(self, discriminator_cost) -> None:
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        # tf.train.AdamOptimizer().minimize(discriminator_cost, var_list=discriminator_parameters)
        return  tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize( -1*discriminator_cost, var_list=discriminator_parameters)


    def clip_parameters(self, fixed_value:float):
        #Try a small value  like 0.01
        discriminator_parameters = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]
        clip_D = [p.assign(tf.clip_by_value(p, -fixed_value, fixed_value)) for p in discriminator_parameters]
        return clip_D
