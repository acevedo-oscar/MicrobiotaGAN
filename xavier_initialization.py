import tensorflow as tf

def xavier_init(size):
    print("my size is "+str(size))
    print(type(size))
    print(len(size))

    if size  == 1:
        size.append(None)
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
