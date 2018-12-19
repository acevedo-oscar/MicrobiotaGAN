import tensorflow as tf
from MicrobiotaGAN.cost import wasserstein_discriminator_cost
from MicrobiotaGAN.cost import wasserstein_generator_cost


def train_graph(my_discriminator, my_generator, n_species, noise_dim):
    # Inputs
    train_real_sample = tf.placeholder(tf.float32, shape=[None, n_species], name="train_real_sample")
    train_noise_sample = tf.placeholder(tf.float32, shape=[None, noise_dim], name="train_noise_sample")

    # Compute Logits
    D_logit_real_train = my_discriminator.train_probability_and_logit(train_real_sample)

    fake_gan_samples = my_generator.train_draw_samples(train_noise_sample)
    D_logit_fake_train = my_discriminator.train_probability_and_logit(fake_gan_samples)

    # Compute Cost
    D_cost_train = wasserstein_discriminator_cost(D_logit_real_train, D_logit_fake_train)
    G_cost_train = wasserstein_generator_cost(D_logit_fake_train)

    # Define Optimize Step
    D_train_step = my_discriminator.optimize_step(D_cost_train)
    G_train_step = my_generator.optimize_step(G_cost_train)

    # Others
    clip_D = my_discriminator.clip_parameters(0.01)

    graph_saver = tf.train.Saver()

    return [train_real_sample, train_noise_sample, G_cost_train, G_train_step, D_cost_train, D_train_step, clip_D,
            D_logit_real_train, D_logit_fake_train, graph_saver]


def inference_graph(my_discriminator, my_generator, n_species, noise_dim):
    # Inputs
    inference_real_sample = tf.placeholder(tf.float32, shape=[None, n_species])
    inference_noise_sample = tf.placeholder(tf.float32, shape=[None, noise_dim])

    # Compute Logits
    D_logit_real_inference = my_discriminator.inference_probability_and_logit(inference_real_sample)

    generator_sample_inference = my_generator.inference_draw_samples(inference_noise_sample)
    D_logit_fake_inference = my_discriminator.inference_probability_and_logit(generator_sample_inference)

    D_cost_inference = wasserstein_discriminator_cost(D_logit_real_inference, D_logit_fake_inference)
    G_cost_inference = wasserstein_generator_cost(D_logit_fake_inference)

    # Others
    clip_D = my_discriminator.clip_parameters(0.01)

    graph_saver = tf.train.Saver()

    return [inference_real_sample, inference_noise_sample, G_cost_inference, D_cost_inference, clip_D, graph_saver]


