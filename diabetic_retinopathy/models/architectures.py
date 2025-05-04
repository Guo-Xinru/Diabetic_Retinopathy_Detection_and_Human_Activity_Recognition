import gin
import tensorflow as tf
from models.layers import vgg_block, residual_block

@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'
    print('Insgesamt ' + str(n_blocks) + ' layers')
    inputs = tf.keras.Input(input_shape)
    out = residual_block(inputs, base_filters, name='first') # This is the first vgg block
    for i in range(2, n_blocks):
        if i == n_blocks - 1:
            out = residual_block(out, base_filters * 2 ** (i), name='last')
        else:
            out = residual_block(out, base_filters * 2 ** (i), name= str(i)) # The second and third block
        print(base_filters * 2 ** (i))

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')