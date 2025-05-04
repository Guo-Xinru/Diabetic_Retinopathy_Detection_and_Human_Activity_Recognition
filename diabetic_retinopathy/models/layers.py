import gin
import tensorflow as tf

@gin.configurable
def vgg_block(inputs, filters, kernel_size, name):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.leaky_relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.leaky_relu, name=name)(out)
    # out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out


"""
rsidual_block to reaplace the vgg_block to do the convolution and add the output
add l2 regularization to the convolutional layer
"""
@gin.configurable
def residual_block(inputs, filters, kernel_size, strides=(1, 1), name='residual_block'):

    residual = inputs
    print("intput shape:", inputs.shape)
    out = tf.keras.layers.Conv2D(filters, kernel_size,
                                 padding='same',
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU(name=f"{name}_relu1")(out)
    out = tf.keras.layers.Conv2D(filters, kernel_size,
                                 padding='same',
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=0.01), name=name)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    if residual.shape[-1] != out.shape[-1]:
        residual = tf.keras.layers.Conv2D(out.shape[-1], (1, 1), strides=strides, padding='same')(residual)
        residual = tf.keras.layers.MaxPool2D((2, 2))(residual)
        print("out shape:", residual.shape)

    out = tf.keras.layers.Add()([out, residual])
    out = tf.keras.layers.ReLU()(out)

    return out
