import tensorflow as tf

from MicrobiotaGAN.xavier_initialization import xavier_init

class Generator:

    def __init__(self,noise_dim : int ,n_species: int) -> None:
         
        nodes_input_layer : int = 128

        self.G_W1 = tf.Variable(xavier_init([noise_dim, nodes_input_layer]) , name="G_W1")
        self.G_b1 = tf.Variable(tf.zeros(shape=[nodes_input_layer]), name="G_b1" )

        self.G_W2 = tf.Variable(xavier_init([nodes_input_layer, n_species ]) , name="G_W2")
        self.G_b2 = tf.Variable(tf.zeros(shape=[n_species]) , name="G_b2")

    def  draw_samples(self, noise, train_bool: bool = True ) :
        

        G_h1 = tf.nn.relu(tf.matmul(noise, self.G_W1) + self.G_b1)
        #norm_G_h1 = tf.layers.batch_normalization(G_h1, training=train_bool)

        mean_G_h1, var_G_h1 = tf.nn.moments(G_h1, axes = [0])
        norm_G_h1 = tf.nn.batch_normalization(G_h1, mean=mean_G_h1,
                                             variance=var_G_h1, offset=0.0,
                                             scale = 1.0, variance_epsilon= 1e-8 )

        G_log_prob = tf.matmul(norm_G_h1, self.G_W2) + self.G_b2
        #norm_G_log_prob = tf.layers.batch_normalization(G_log_prob, training=train_bool)

        #mean_G_log_prob, var_G_log_prob = tf.nn.moments(G_h1, axes = [0])
        """
        norm_G_log_prob = tf.nn.batch_normalization(G_log_prob, mean=mean_G_log_prob,
                                             variance=var_G_log_prob, offset=0.0,
                                             scale = 1.0, variance_epsilon= 1e-8 )

        """
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    """
    def draw_samples_inference(self, noise ) :

        train_bool = False

        G_h1 = tf.nn.relu(tf.matmul(noise, self.G_W1) + self.G_b1)
        norm_G_h1 = tf.layers.batch_normalization(G_h1, training=train_bool)


        G_log_prob = tf.matmul(norm_G_h1, self.G_W2) + self.G_b2
        norm_G_log_prob = tf.layers.batch_normalization(G_log_prob, training=train_bool)


        G_prob = tf.nn.sigmoid(norm_G_log_prob)

        return G_prob
    """

    def optimize_step(self, generator_cost) -> None:
        generator_parameters = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        # tf.train.AdamOptimizer().minimize(generator_cost, var_list=generator_parameters)
        return  tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(generator_cost, var_list=generator_parameters)
