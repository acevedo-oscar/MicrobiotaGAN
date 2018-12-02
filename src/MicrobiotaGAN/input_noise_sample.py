import numpy as np

def input_noise_sample(mini_batch_size : int, nodes_generator_input_layer : int):
    return np.random.uniform(-1., 1., size=[mini_batch_size, nodes_generator_input_layer]).astype(np.float32)
