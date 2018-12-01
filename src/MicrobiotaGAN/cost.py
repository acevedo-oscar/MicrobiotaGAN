import numpy as np
import tensorflow as tf


def wasserstein_generator_cost( logit_fake_sample):
    # tf.reduce_mean computes the mean along the columns or row of your tensor
    G_cost = -tf.reduce_mean(logit_fake_sample)

    return G_cost

def wasserstein_discriminator_cost(logit_real_sample, logit_fake_sample):
    return  tf.reduce_mean(logit_real_sample) - tf.reduce_mean(logit_fake_sample)



def generator_cost( logit_fake_sample):
    # We use logit_fake_sample because the Generator only cares how good of a job it did in fooling the discriminator
    G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake_sample, labels=tf.ones_like(logit_fake_sample))

    # Compute cost by computing the mean of the losses
    # tf.reduce_mean computes the mean along the columns or row of your tensor
    G_cost = tf.reduce_mean(G_loss)

    return G_cost

def discriminator_cost(logit_real_sample, logit_fake_sample):
    # Define losses
    D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real_sample, labels=tf.ones_like(logit_real_sample))
    D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake_sample, labels=tf.zeros_like(logit_fake_sample))

    # Compute cost by computing the mean of the losses
    # tf.reduce_mean computes the mean along the columns or row of your tensor
    D_cost_real = tf.reduce_mean(D_loss_real)
    D_cost_fake = tf.reduce_mean(D_loss_fake)

    Total_Discriminator_Cost = D_cost_real + D_cost_fake

    return Total_Discriminator_Cost
