
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared

def prepare_input(X, y):
    shared_x = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
    return shared_x, theano.tensor.cast(shared_y, "int32")

def one_hot_encoder(data):
    """
    Converts target into one-hot-encoder for input
    """
    create_entry = lambda x : [1, 0] if x == 0 else [0, 1]
    data1 = []
    for x in data:
        e = create_entry(x)
        data1.append(e)
    return np.array(data1)

def load_mnist_data(filename=None):
    if filename == None:
        f = gzip.open('mnist.pkl.gz', 'rb')
    else:
        f = gzip.open(filename, 'rb')
    
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()

    def shared(data):
        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, theano.tensor.cast(shared_y, "int32")
    
    return [shared(training_data), shared(validation_data), shared(test_data)]
