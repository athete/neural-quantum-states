import tensorflow as tf
from tensorflow.keras.layers import Layer 

class BiasLayer(Layer):
    """Creates a layer with only a trainable bias term.
    """
    def __init__(self):
        super(BiasLayer, self).__init__()
    
    def build(self, input_shape):
        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            initializer=tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005),
            trainable=True 
        )

    def call(self, inputs):
        return inputs + self.bias