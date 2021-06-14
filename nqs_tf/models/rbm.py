import tensorflow as tf
from .layers import BiasLayer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class RBM(Model):
    """Creates a simple RBM architecure
    """
    def __init__(self, num_hidden, name='RBM'):
        super(RBM, self).__init__(name=name)
        self.visible_layer = BiasLayer()
        self.hidden_layer = Dense(
            units=num_hidden, 
            activation=tf.keras.activations.linear, 
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005),
            name='hidden_layer'
        )
    
    def call(self, inputs):
        x = self.visible_layer(inputs)
        x = self.hidden_layer(x)
        return tf.reduce_sum(x, axis=0)