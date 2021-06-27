import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from activations.activations import tan_sigmoid, exponential

class FFNN(Model):
    """ Creates a generic Feedforward neural network.
    """
    def __init__(self):
        super(FFNN, self).__init__()
    
    def build(self, input_shape):
        self.dense1 = Dense(units=input_shape[-1], activation=tan_sigmoid)
        self.output_layer = Dense(units=1, activation=exponential)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.output_layer(x)
        return tf.reduce_sum(x, axis=-1)

