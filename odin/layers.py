"""
http://neuralnetworksanddeeplearning.com/chap6.html
"""

#### Libraries
# Standard library
import sys
import random

# Third-party libraries
import numpy as np
import theano
import theano.tensor.signal.pool
import theano.tensor as T

class convolutionpoollayer(object):

    def __init__(self, filter_shape, image_shape, poolsize=(2,2),
                 activation_fn = T.nnet.sigmoid):
        """
        `filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn

        n_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize)
        
        self.w = theano.shared(np.asarray(np.random.normal(
                                          loc=0.0,
                                          scale=np.sqrt(1.0/n_out),
                                          size=filter_shape), dtype=theano.config.floatX),
                                name='w',
                                borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        
        self.params = [self.w, self.b]
    

    def _set_input(self, _input, mini_batch_size):    
        self._input = _input.reshape(self.image_shape)
        conv_out = T.nnet.conv.conv2d(
                   input=self._input, filters=self.w, filter_shape=self.filter_shape,
                   image_shape=self.image_shape)
        pooled_out = theano.tensor.signal.pool.pool_2d(
                     input=conv_out, ws=self.poolsize, ignore_border=True)
        self._output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class fullyconnectedlayer(object):
    """ Class that implements fully connected layer """

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = T.nnet.sigmoid

        # initialize weights
        self.w = theano.shared(np.asarray(np.random.normal(
                                          loc=0.0,
                                          scale=np.sqrt(1.0/n_out),
                                          size=(n_in, n_out)), dtype=theano.config.floatX),
                                name='w', 
                                borrow=True)
        self.b = theano.shared(np.asarray(np.ones((n_out,)), dtype=theano.config.floatX),
                                name='b', borrow=True)
        
        self.params = [self.w, self.b]
    
    def _set_input(self, _input, mini_batch_size):
        self._input = _input.reshape((mini_batch_size, self.n_in))
        self._output = self.activation_fn(T.dot(self._input, self.w) + self.b)
        self.y_out = T.argmax(self._output, axis=1)
    
    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return T.mean(T.eq(y, self.y_out))


class softmaxlayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        # initialize weights and baises
        self.w = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), 
                               name='w',
                               borrow=True)
        
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)
        
        self.params = [self.w, self.b]
    
    def _set_input(self, _input, mini_batch_size):
        self._input = _input.reshape((mini_batch_size, self.n_in))
        self._output =  T.nnet.softmax(T.dot(self._input, self.w) + self.b)
        self.y_out = T.argmax(self._output, axis=1)
    
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self._output)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))