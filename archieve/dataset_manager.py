import numpy
def normalize_ds(dataset):
    """
    Normalizes using the following formula. We use it because species abundance
    should be greater or equal to zero.

    Z = X - min(X) / max(X) - min(x)

    It assumes that the dataset is two-dimentional matrix where each row represents
    a different community.

    Outliers may cause troubles
    """
    dataset = copy.copy(dataset)

    dim_dataset = dataset.shape

    for n_row in range(dim_dataset[0]):
        k = dataset[n_row,:]
        k_norm =(k - np.min(k))/(np.max(k) - np.min(k))
        dataset[n_row,:] = k_norm

    return dataset


class DataSetManager:

    def __init__(self, images_set, norm=False) -> None:

        if norm:
            ds = normalize_ds(images_set)
        else:
            ds = images_set

        self._images = ds
        self.num_examples = images_set.shape[0]
        # self._epochs_completed : int = 0
        self.epochs_completed: int = 0
        self._index_in_epoch = 0


    @property
    def images(self):
        return self._images

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self.num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]

        # Go to the next epoch
        if start + batch_size > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start
            images_rest_part = self._images[start:self.num_examples]
             # Shuffle the data
            if shuffle:
                perm = numpy.arange(self.num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]

            return numpy.concatenate( (images_rest_part, images_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end]
