
# 3rd party libraries
import numpy as np

import theano
import theano.tensor as T
from theano import function
from theano import shared

import matplotlib.pyplot as plt

# Odin library classes
import network
import helpers
from layers import convolutionpoollayer, fullyconnectedlayer, softmaxlayer

def main():
    """ Main """
    training_data, validation_data, test_data = helpers.load_mnist_data("odin/datasets/mnist.pkl.gz")
    net = network.network([
          convolutionpoollayer(filter_shape=(20, 1, 5, 5), image_shape=(10, 1, 28, 28), poolsize=(2,2)),
          fullyconnectedlayer(n_in=12*12*20, n_out=100),
          softmaxlayer(n_in=100, n_out=10)],
          mini_batch_size=10)
    net.fit(training_data, 60, mini_batch_size=10, eta=0.1, test_data=test_data, validation_data=validation_data)

if __name__ == "__main__":
    main()