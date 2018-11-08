import numpy as np
from MicrobiotaGAN.display_microbiota import display_microbiota

def show_rounded_array(my_array, decimal -> int):
    print( [ round(my_array[k],decimal) for k in range(len(my_array)) ])

def get_gan_samples_image(generator):
    """
    Generates an 28x28 images from trained GAN.
    """
    virtual_microbiota = []
    for k in range(28):
        virtual_microbiota.append( sess.run(generator.draw_samples(input_noise_sample(1,10)))[0])
    virtual_microbiota = np.array(virtual_microbiota)
    display_microbiota(virtual_microbiota)

def unroll_dataset(ds):
    ds = np.array(ds)
    dim_ds = ds.shape
    unrolled_ds = []

    for n_images in range(dim_ds[0]):
        for n_row in range(dim_ds[1]):
            # add all species from community n_row from n_image
            unrolled_ds.append(ds[n_images,n_row,:])
    return np.array(unrolled_ds)

def ds_statistic_summary_dict(ds):
    """
    Computes a vector for the mean and variance of each row, and the mean and
    variance of this. Also the max and means of each row.

    These might be useful to scale back the data.
    """
    ds = unroll_dataset(ds)
    n_rows = ds.shape[0]

    # axis = 1 computes the mean of each row
    ds_vec_mean = np.mean(ds, axis=1)

    np.std(ds[0], ddof=1)

    """
    ddof stands for Delta Degrees of Freedom

    We use ddof=1 because we are  estimating one statistic from an estimate
    of another. Namely, the mean.
    """
    ds_vec_std = np.std(ds, ddof=1, axis=1)

    # Compute min and max, which are used to normalize in our DS manager
    max_vec = []
    for k in range(n_rows):
        max_vec.append(np.max(ds[k,:]))
    max_vec = np.array(max_vec)

    min_vec = []
    for k in range(n_rows):
        min_vec.append(np.min(ds[k,:]))
    min_vec = np.array(min_vec)

    ds_dict = {"ds_vec_mean":ds_vec_mean, "ds_vec_std", "max_vec":max_vec, "min_vec":min_vec  }

    return ds_dict
