import numpy as np
import tensorflow as tf

def generateRandomData(num_samples, num_dimensions, num_categories):
    data = np.random.random((num_samples, num_dimensions))
    labels = np.random.random((num_samples, num_categories))
    return data, labels

def loadToyData(data, labels, val_data, val_labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(32).repeat()

    return dataset, val_dataset
