from timeit import default_timer as timer
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
working_dir= os.getcwd()
os.chdir("../src")
# === Import my library module
from MicrobiotaGAN.dataset_manager import DataSetManager
from MicrobiotaGAN.utilities import *
from MicrobiotaGAN.input_noise_sample import input_noise_sample
from MicrobiotaGAN.glv_loss import GLV_Model
os.chdir(working_dir)


def pseudo_log_transformation(data_table):
    """
    if np.sum(data_table < -0.001) == 0:
        print("There's not a negative number with big magnitude")
    """
    if np.sum(data_table < 0) > 0:
        # print("negative value exists")

        data_table[data_table < 0] = 1.0e-7
        data_table[data_table == 0] = 1.0e-7

    data_table = np.log10(data_table)
    data_table[data_table == -np.inf ] = -6

    return data_table


def inverse_pseudo_log_transformation(data_table):
    return np.power(10, data_table)


def load_data():
    raw_microbiota_table = pd.read_csv('../data/abundances.csv', header=None).values

    param_path = '../data/image_dynamics_param.pickle'

    with open(param_path, 'rb') as pickleFile:
        microbiota_parameters = pickle.load(pickleFile)

    # ===> DATA TRANSFORMATION <====
    microbiota_table = raw_microbiota_table[0:10000]

    microbiota_table = pseudo_log_transformation(microbiota_table)
    # ===> DATA TRANSFORMATION <====

    """
    print("====> DEBUGGING <====")
    print(np.max(microbiota_table))
    print(np.min(microbiota_table))
    print("\tAGAINST")
    print(np.max(raw_microbiota_table[0:10000]))
    print(np.min(raw_microbiota_table[0:10000]))
    """




    # microbiota_table = np.multiply(microbiota_table, 0.1)

    np.random.shuffle(microbiota_table)

    training_percent = 0.9

    lim = int(np.ceil(microbiota_table.shape[0] * training_percent))
    train_set = microbiota_table[0:lim, :]
    test_set = microbiota_table[lim:, :]
    print("Test set" + str(test_set.shape))
    print("Train set: " + str(train_set.shape))

    microbiota_test_set = DataSetManager(test_set, norm=False)
    microbiota_train_set = DataSetManager(train_set, norm=False)

    m_sigma = microbiota_parameters['sigma']
    m_rho = microbiota_parameters['rho']

    print("\n")
    print("Sigma: " + str(m_sigma))
    print("Rho: " + str(m_rho))

    return [microbiota_table, microbiota_test_set, microbiota_train_set, microbiota_parameters]


""" 
m_r = microbiota_parameters['r']

# n_species = m_A.shape[0]

# global n_species  = m_A.shape[0]
# global mini_batch_size: int = 32    
m_A = microbiota_parameters['A']
"""
def train_gan(microbiota_train_set, graph_train_operations, mini_batch_size,
              train_epochs, my_discriminator, my_generator, text_log=False):
    noise_dim = 10

    print("Starting Traning Loop \n")
    train_real_sample, train_noise_sample ,D_cost_train, clip_D, D_train_step, G_train_step, G_cost_train, train_graph_saver = graph_train_operations

    # number_of_fig_per_plot: int = 16
    # Training Loop
    counter = 0
    d_iter_ratio: int = 5

    # train_epochs = 100

    iterations_per_epoch = microbiota_train_set.num_examples//mini_batch_size

    iteration_number = train_epochs * iterations_per_epoch
    # Since the discriminator is trained 5 times more, de divide the number of iterations
    iteration_number = int(np.ceil(iteration_number / d_iter_ratio))

    start = timer()

    train_g_cost_record = []
    train_d_cost_record = []
    iter_record_g = []
    iter_record_d = []

    epoch_record_g = []
    epoch = []

    g_train_epoch_cost = []
    d_train_epoch_cost = []

    initial_epochs = microbiota_train_set.epochs_completed

    little_record = []

    add_g_record = 0
    D_current_cost = 0
    G_current_cost = 0

    gen_logit = []
    dis_logit = []

    #### Sess 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        it = 0
        while microbiota_train_set.epochs_completed - initial_epochs < train_epochs:

            it += 1
            """
              if microbiota_train_set.epochs_completed == 2:
                little_record.append(sess.run(my_generator.G_W1))
          
            """

            # Train more the discriminator
            for k in range(d_iter_ratio):

                previous_epoch = microbiota_train_set.epochs_completed
                real_sample_mini_batch = microbiota_train_set.next_batch(mini_batch_size)
                current_epoch = microbiota_train_set.epochs_completed

                noisy1 = input_noise_sample(mini_batch_size, noise_dim)

                """
                print("Discriminator data")
                real_string = str(np.sum(real_sample_mini_batch))+" "+str(real_sample_mini_batch.shape) 
                fale_string = str(np.sum(noisy1))+" "+str(noisy1.shape) 
                print("Real "+real_string+" fake "+fale_string)
                """

                dis_var_dict = {train_real_sample: real_sample_mini_batch,
                                train_noise_sample: noisy1
                                }

                D_train_step.run(feed_dict=dis_var_dict)
                D_current_cost, _ = sess.run([D_cost_train, clip_D], feed_dict=dis_var_dict)

                # ======> Debug logits <===========
                # dis_logit.append(sess.run([D_logit_fake_train], feed_dict=dis_var_dict))
                # gen_logit.append(sess.run([D_logit_real_train], feed_dict=dis_var_dict))
                # dis_logit.append(sess.run([norm_logit_t_fake,  D_logit_fake_train, fake_L1_output ], feed_dict=dis_var_dict))
                # gen_logit.append(sess.run([norm_logit_t_real,  D_logit_real_train,  real_L1_output], feed_dict=dis_var_dict))
                # ======> Debug logits <===========

                train_d_cost_record.append(D_current_cost)
                iter_record_d.append(it + 1 + k)

                if current_epoch > previous_epoch:
                    d_train_epoch_cost.append(D_current_cost)
                    add_g_record = 1
                    # g_train_epoch_cost.append(G_current_cost)

            # End For

            noisy2 = input_noise_sample(mini_batch_size, noise_dim)
            # Apply the optimization algorithm and update both network parameters
            gen_var_dict = {train_noise_sample: noisy2}
            # _, G_current_cost = sess.run([G_train_step, G_cost_train], feed_dict=gen_var_dict)
            G_train_step.run(feed_dict=gen_var_dict)

            G_current_cost = sess.run([G_cost_train], feed_dict=gen_var_dict)

            # print(G_current_cost)
            # print("noise: "+str(np.sum(noisy)))

            G_current_cost = G_current_cost[0]

            train_g_cost_record.append(G_current_cost)

            if add_g_record == 1:
                g_train_epoch_cost.append(G_current_cost)
                add_g_record = 0

            iter_record_g.append(it)

            if text_log:

                if it % (iteration_number // 10) == 0 or it == 0:
                    end = timer()
                    print("Progreso :" + str(round(100 * it / (1.0 * iteration_number))) + " %")
                    print(str(end - start) + " s")

                    current_gen_weights = np.sum(sess.run(my_generator.G_W1))
                    current_dis_weights = np.sum(sess.run(my_discriminator.D_W1))

                    # print("\t Current Gen Weights " +str(current_gen_weights))
                    # print("\t Current Dis Weights " +str(current_dis_weights))

                    print("Epochs completed so far " + str(microbiota_train_set.epochs_completed))

                    print('\t Iter: {}'.format(it))
                    print('\t D loss: {:.4}'.format(D_current_cost))
                    print('\t G_loss: {:.4}'.format(G_current_cost))
                    print("\n")

                """
                print(len(dis_logit[-1][0]))
                print(type(dis_logit[-1]))
                print(np.sum(dis_logit[-1]))
                print(np.asarray(dis_logit[-1]).shape)
                print(np.sum(gen_logit[-1]))

                print(np.asarray(gen_logit[-1]).shape)
                print(np.mean(gen_logit[-1]))           



                print("Norm Logit Fake \t"+str(np.sum(dis_logit[-1][0])))
                print("Logit Fake \t"+str(np.sum(dis_logit[-1][1])))
                print("L1 Output Fake \t"+str(np.sum(dis_logit[-1][2])))

                print("Norm Logit Real \t"+str(np.sum(gen_logit[-1][0])))
                print("Logit Real \t"+str(np.sum(gen_logit[-1][1])))
                print("L1 Output Real \t"+str(np.sum(gen_logit[-1][2])))


                """

        train_graph_saver.save(sess, '../results/trained_gan.ckpt')
        little_record.append(sess.run(my_generator.G_W1))

    return [d_train_epoch_cost, g_train_epoch_cost]
    # epochs_global += train_epochs


def draw_gan_samples(my_generator, number_of_samples_to_draw=1000, noise_dim=10):


    input_noise = tf.placeholder(tf.float32, shape=[None, noise_dim])
    get_sample = my_generator.inference_draw_samples(input_noise)

    sample_holder = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '../results/trained_gan.ckpt')

        input_dict = {input_noise: input_noise_sample(number_of_samples_to_draw, noise_dim)}

        sample_holder.append(sess.run([get_sample], feed_dict=input_dict))

    return sample_holder[0][0]


def test_gan(microbiota_test_set, test_train_operations, mini_batch_size, train_epochs, noise_dim=10, text_log=False):

    inference_real_sample, inference_noise_sample, D_cost_inference, clip_D, G_cost_inference, test_graph_saver= test_train_operations


    number_of_fig_per_plot: int = 16
    # Training Loop
    counter = 0

    test_epochs = train_epochs

    iterations_per_epoch = microbiota_test_set.num_examples // mini_batch_size

    test_iter = test_epochs * iterations_per_epoch

    # test_iter =  (numero_de_iteraciones*test_set.shape[0])//train_set.shape[0]

    start = timer()

    test_g_cost_record = []
    test_d_cost_record = []
    iter_test_record_g = []
    iter_test_record_d = []

    g_test_epoch_cost = []
    d_test_epoch_cost = []

    epoch_record_g = []
    epoch = []

    d_iter_ratio: int = 5

    """
    print(iterations_per_epoch)
    print(test_iter)
    print(test_set.shape[0])

    """

    #saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_graph_saver.restore(sess, '../results/trained_gan.ckpt')
        print("\n")

        for it in range(test_iter):

            # Train more the discrimantor
            for k in range(d_iter_ratio):
                # f k%5 == 0 and k != 0:
                # pass

                real_sample_mini_batch = microbiota_test_set.next_batch(mini_batch_size)
                dis_var_dict = {inference_real_sample: real_sample_mini_batch,
                                inference_noise_sample: input_noise_sample(mini_batch_size, noise_dim)
                                }
                D_current_test_cost, _ = sess.run([D_cost_inference, clip_D], feed_dict=dis_var_dict)
                test_d_cost_record.append(D_current_test_cost)
                iter_test_record_d.append(it + 1 + k)

            # End For

            # Apply the optimization algorithm and update both newtwork parameters
            gen_var_dict = {inference_noise_sample: input_noise_sample(mini_batch_size, noise_dim)}
            temp_g_cost = sess.run([G_cost_inference], feed_dict=gen_var_dict)
            # Since is the single fetch
            G_current_test_cost = temp_g_cost[0]

            test_g_cost_record.append(G_current_test_cost)

            iter_test_record_g.append(it)

            if it % iterations_per_epoch == 0 and (k != 0):
                d_test_epoch_cost.append(D_current_test_cost)
                g_test_epoch_cost.append(G_current_test_cost)

            if text_log:
                if it % (test_iter // 10) == 0:
                    end = timer()
                    print("Progreso :" + str(round(100 * it / (1.0 * test_iter))) + " %")
                    print(str(end - start) + " s")

                    print("epochs completed " + str(microbiota_test_set.epochs_completed))
                    print(it / (iterations_per_epoch / 5))

                    print('\t Iter: {}'.format(it))
                    print('\t D loss: {:.4}'.format(D_current_test_cost))
                    print('\t G_loss: {:.4}'.format(G_current_test_cost))

    return [d_test_epoch_cost, g_test_epoch_cost ]


def plot_cost(train_cost, test_cost, label_description: str):
    # # Generator Loss Graph

    g_test_epoch_cost = train_cost
    g_train_epoch_cost = test_cost

    epochs_list = list(range(len(g_test_epoch_cost)))

    plt.plot(epochs_list, g_train_epoch_cost)
    plt.plot(epochs_list, g_test_epoch_cost)
    plt.legend(['train set', 'test set'], loc='upper right')

    plt.xlabel("Epochs")
    plt.ylabel("Epoch Cost");
    plt.title(label_description+" Cost per Epoch")

    im_ratio = 18.5 / 10.5

    fig = plt.gcf()
    fig.set_size_inches(12, 12 / im_ratio)

    plt.savefig('../results/generator_loss.png', dpi=300)


def glv_loss(samples, m_A, m_R):

    glv_errors = []
    for k in range(samples.shape[0]):
        error = np.sum(GLV_Model(samples[k, :], m_A, m_R))
        glv_errors.append(error)

    glv_errors = np.array(glv_errors)

    return glv_errors


def plot_glv_error_boxplot(glv_gen_errors, glv_train_errors, train_epochs:int , n_samples:int):
    d = {'GAN': glv_gen_errors, 'Training': glv_train_errors}
    df = pd.DataFrame(data=d)

    sns.boxplot(data=df)

    plt.title("GLV Error, " + str(train_epochs) + " epochs , " + str(n_samples) + " samples draw")

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.savefig("glv_barplot.png")


def gan_glv_cost(my_generator, m_A, m_r, n_samples = 10000):
    # Draw samples form the GAN

    gan_samples = draw_gan_samples(my_generator, number_of_samples_to_draw=n_samples)
    gan_samples = inverse_pseudo_log_transformation(gan_samples)

    gan_samples_error = glv_loss(gan_samples, m_A, m_r)

    return gan_samples_error


def standard_error_of_the_mean(data_table):
    return np.std(data_table)/np.sqrt(data_table.shape[0])


def plot_glv_epoch_standard_error(epochs, errors):

    constant_line = np.mean(errors)*np.ones(len(epochs))

    plt.plot(epochs,errors)
    plt.plot(epochs, constant_line)

    plt.xlabel("Epoch")
    plt.ylabel("GLV STD error of the mean");
    plt.title("GLV STD error of the mean")
    plt.legend(['GLV STD error of the mean', 'mean'], loc='upper right')

    im_ratio = 18.5 / 10.5

    fig = plt.gcf()
    fig.set_size_inches(12, 12 / im_ratio)

    plt.savefig('../results/glv_std_error.png', dpi=300)


def plot_glv_cost(epochs, errors):

    constant_line = np.mean(errors)*np.ones(len(epochs))
    plt.plot(epochs, errors)
    plt.plot(epochs, constant_line)

    plt.xlabel("Epoch")
    plt.ylabel("GLV Epoch Cost");
    plt.title("GLV Cost per epoch")
    plt.legend(['GLV Cost per epoch', 'mean'], loc='upper right')

    im_ratio = 18.5 / 10.5

    fig = plt.gcf()
    fig.set_size_inches(12, 12 / im_ratio)

    plt.savefig('../results/glv_cost per epoch.png', dpi=300)