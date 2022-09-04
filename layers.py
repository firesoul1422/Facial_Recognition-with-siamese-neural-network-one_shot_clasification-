# Custom L1 distance layer module
# when you do custom object in tensorflow you must bring it with the model

import tensorflow as tf
from tensorflow.keras.layers import Layer


# Custom layer from jupyter notebook
class L1Diist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    # this function activated when data passed to the layer - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

