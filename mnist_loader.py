"""

.pkl extension files are Pickled files using the `pickle` python module.
We can load a compressed file during runtime!
"""
from os import path
import pickle
import gzip

import numpy as np

def load_data():
    """
    Returns MNIST data as a tuple of
      (training_data, validation_data, test_data)
    
    training_data: is a duple, 
      (training_images in numpy arr form, training_labels as int values)
      len=50,000

    validation_data: same shape,
       len=10,000
    
    test_data: same shape,
       len=10,000
    """

    MNIST_PATH = './data/mnist.pkl.gz'
    if path.exists(MNIST_PATH):
        f = gzip.open(MNIST_PATH, 'rb')
        train, validation, test = pickle.load(f, encoding='latin1')
        f.close()

        return train, validation, test
    else:
        print(f"File {MNIST_PATH} not found")
        exit()

def vectorized_result(j):
    """
    Returns a 10-dimensional unit vector, with the value
    of 1.0 in the given j position, zeros elsewhere.
    Converts a digit into the corresponding expected output
    from the neural ShallowNetwork of 10-neuron output layer
    """
    e = np.zeros((10, 1)) # column vector length 10
    e[j] = 1.0
    return e

def load_data_wrapper():
    """
    Format the result of `load_data` more conveniently
    for neural ShallowNetwork operation.
    """
    tr_d, va_d, te_d = load_data()

    training_x = [np.reshape(x, (784, 1)) for x in tr_d[0]]  # flatten to a 784 entry column vector
    training_y = [vectorized_result(y) for y in tr_d[1]]
    training_data = [(x, y) for (x,y) in zip(training_x, training_y)]

    # For validation and test sets we do not transform label digit into vector form
    validation_x = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = [(x, y) for (x,y) in zip(validation_x, va_d[1])]

    test_x = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = [(x, y) for (x,y) in zip(test_x, te_d[1])]

    return (training_data, validation_data, test_data)