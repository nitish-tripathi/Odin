
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
    training_data, validation_data, test_data = helpers.load_mnist_shared_data("odin/datasets/mnist.pkl.gz")
    
    X, y = training_data
    
    X_test = theano.shared(np.asarray(X.get_value()[0:3], dtype=theano.config.floatX), borrow=True)
    y_test = theano.tensor.cast(theano.shared(np.asarray(y.eval()[0:3], dtype=theano.config.floatX), borrow=True), "int32")

    net = network.network([
          convolutionpoollayer(filter_shape=(20, 1, 5, 5), image_shape=(10, 1, 28, 28), poolsize=(2,2)),
          fullyconnectedlayer(n_in=12*12*20, n_out=100),
          softmaxlayer(n_in=100, n_out=10)],
          mini_batch_size=10, epochs=5, eta=0.1)
    
    net.fit(X, y)
    a = net.evaluate(X, y)
    print a

if __name__ == "__main__":
    main()