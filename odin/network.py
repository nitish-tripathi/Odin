"""
# Nitish Tripathi 2017
# odin Deep Learning Library
#
# Implements base network class for developing neural network
# Author: Nitish Tripathi
#
# License: MIT
"""

#### Libraries
# Standard library
import sys
import random
import json

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T

# Odin libraries
from layers import convolutionpoollayer, fullyconnectedlayer, softmaxlayer

class network(object):
    def __init__(self, layers, mini_batch_size, epochs, eta, l2=0.0):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = eta
        self.l2 = l2

        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        init_layer = self.layers[0]
        init_layer._set_input(self.x, mini_batch_size)

        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer._set_input(prev_layer._output, self.mini_batch_size)
        
        self._output = self.layers[-1]._output
    
    def evaluate(self, X, y):
        """Evaluate the result """

        input_size = X.shape.eval()[0]
        num_test_batches = input_size/self.mini_batch_size
        
        i = T.lscalar()
        evaluate_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: 
                X[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        result = np.mean(
                    [evaluate_accuracy(j) for j in xrange(num_test_batches)])
        
        return result
       
    def fit(self, X, y):
        """Train the network using mini-batch stochastic gradient descent."""

        # compute number of minibatches for training, validation and testing
        input_size = X.shape.eval()[0]
        num_training_batches = input_size/self.mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*self.l2*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-self.eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                X[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
            
        # Do the actual training
        for epoch in xrange(self.epochs):
            print("Training epoch number: {0}".format(epoch))
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                cost_ij = train_mb(minibatch_index)

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]