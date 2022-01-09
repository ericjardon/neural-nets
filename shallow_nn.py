"""
Module that implements a basic feedforward ShallowNetwork of
sigmoid neurons.

ShallowNetwork learns using Stochastic Gradient Descent, using
backpropagationa algorithm to compute the gradients of the Cost function
for a particular observation x.

NOTE: this module is not meant to be performant in production, but rather for learning purposes, 
and so is not optimized.
"""

from typing import List, Tuple
import numpy as np
from numpy.core.numerictypes import nbytes
from numpy.random import randn
from numpy.typing import NDArray
# to generate sample from a standard normal distribution, array with shape (d0, d1,...)

import random

# Type aliases
Vector = NDArray 
LabeledDataset = List[Tuple[Vector, int]]

class ShallowNetwork():
    "Implements the most basic neural net"

    def __init__(self, layer_sizes: List[int], verbose=True):
        self.n_layers = len(layer_sizes)
        self.layer_size = layer_sizes
        self.verbose = verbose

        # Initialize weights and biases with random values from a normal

        self.biases = [
            randn(k, 1) for k in layer_sizes[1:]
        ]  # from layer i to i+1, we have layer_size[i+1] biases.

        # weights[0] is a Numpy matrix of the weights connecting layers 0 and 1
        self.weights = [randn(i, j) for j, i in zip(layer_sizes[:-1], layer_sizes[1:])]
        # weights is a matrix axb where a is number of neurons in next layer (rows),
        # and b is number of neurons from previous layer (columns).
        # the output of prev is dot-multiplied with weights for the next, results in col vector with length a


    def feedforward(self, input:Vector):
        """
        Given an input vector to the NN, return the output vector.
        Applies a' = σ(w·a+b) for every layer.
        """
        a = input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)  # vectorized operation

        return a

    def SGD(
        self,
        training_data: LabeledDataset,
        test_data: LabeledDataset,
        epochs: int,
        mini_batch_size: int,
        eta: float,
    ):
        """
        Trains the net using mini-batch stochastic gradient descent.
        training_data and test_data are lists of tuples (x_vect, y)
        test_data is optional.
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):  # how many passes over the training set
            random.shuffle(training_data)
            # split training set into batches of at most mini_batch_size
            batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for batch in batches:
                # For each mini batch, apply a single step of gradient descent:
                # Compute approximate gradient of Cost function
                # and update our weights and biases according to gradient descent
                self.update_nn_weights(batch, eta)

            if self.verbose:
                if test_data:
                    print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
                else:
                    print(f"Epoch {j} completed")

    
    def update_nn_weights(self, mini_batch: LabeledDataset, eta: float):
        """
        Updates the ShallowNetwork's weights and biases.
        Applies gradient descent using backpropagation to a single mini batch.
        mini_batch is a list of tuples (x_vector, y)
        eta is the learning rate (step size in our descent)
        """

        # Initialize gradient approximations
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]


        for x, y in mini_batch:
            # Efficiently compute gradients of every w,b associated to current observation x
            delta_nabla_w, delta_nabla_b= self.backprop(x, y)
            
            # Update gradients of every w and b
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            # Update weights and biases according to their individual gradient: 
            #    vk ← vk - η(dC/dvk)
            self.weights = [
                w - (eta/len(mini_batch))*nw 
                for w, nw in zip(self.weights, nabla_w)
            ] # every w is a 2d numpy array (matrix)
            self.biases = [
                b - (eta/len(mini_batch))*nb
                for b, nb in zip(self.biases, nabla_b)
            ] # every b is a numpy array (vector)

    def backprop(self, x:Vector, y:int):
        """
        Return a tuple `(nabla_b, nabla_w)` representing the
        gradient for the cost function C_x.  
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of 
        numpy arrays, similar to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z) 
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.n_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_w, nabla_b)


    def evaluate(self, test_data:LabeledDataset):
        """
        Return the absolute frequency of correctly labeled 
        test inputs. 
        The output of the NN is the index of whichever neuron in
        the output layer has highest activation
        """
        # prediction is given by np.argmax(output_vector)
        correct = 0
        for (x, y) in test_data:
            pred = np.argmax(self.feedforward(x))
            correct += int(pred == y)
        return correct
    
    def cost_derivative(self, output_activations: Vector, y: Vector):
        """
        Return the vector of partial derivatives
        """
        return output_activations - y  # vectorized operation


### Utility funcs
def sigmoid(z):
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))