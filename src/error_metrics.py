import numpy as np
from scipy.stats import entropy as DKL
from scipy.spatial.distance import jensenshannon as JSD 


def data_probs(ds,bins_partition):
    hist, bin_edges = np.histogram(ds,bins=bins_partition, density=True)
    probs = hist * np.diff(bin_edges)
    return probs

def gan_error(gan_ds, true_ds, error_function):
    """
    Computes the DKL or JSD for 1D data
    """

    # assert gan_ds.ndim == true_ds.ndim
    assert gan_ds.ndim == 1

    partitions= np.linspace(true_ds.min(), true_ds.max(),num=100)

    real_distribution = data_probs(true_ds, partitions)
    estimated_distribution  = data_probs(gan_ds, partitions)

    if error_function == "JSD":
        return JSD(estimated_distribution, real_distribution)


    if error_function == "DKL":
        return DKL(estimated_distribution, real_distribution)
    else:
        print("Invalid error functions")

def gan_error_all_species(gan_ds, true_ds, error_function = "JSD"):
    assert gan_ds.shape[1] == true_ds.shape[1]

    N = gan_ds.shape[1]

    scores  = [gan_error(gan_ds[:, k], true_ds[:,k], error_function) for k in range(N) ]

    return np.mean(scores)
