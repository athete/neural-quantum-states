""" This module implements some commonly used activation functions. """

from tensorflow.keras import backend as K

def tan_sigmoid(z):
    """ Implements the tangent-sigmoid activation function.
    """
    return (2 / (1 + K.exp(-2.0 * z))) - 1

def exponential(z):
    """ Implements the exponential activation function.
    """
    return K.exp(z)

def ReLU(z, alpha=0, max_value=None, threshold=0):
    """ Implements the Rectified Linear Unit (ReLU) activation function.
    """
    return K.relu(z, alpha, max_value, threshold)

def sigmoid(z):
    """ Implements the sigmoid activation function.
    """
    return K.sigmoid(z)