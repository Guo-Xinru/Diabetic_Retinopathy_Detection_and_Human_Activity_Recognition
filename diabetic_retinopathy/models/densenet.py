import gin
import tensorflow as tf


@gin.configurable
#def dense_net(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
def dense_net(input_shape, n_classes, n_blocks, num_layers_per_block, growth_rate, compression):

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'
    print('Insgesamt ' + str(n_blocks) + ' layers')
    inputs = tf.keras.Input(input_shape)

    out = tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=7,
                                 strides=2,
                                 padding='same',
                                 use_bias=False,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding="same")(out)

    for i in range(n_blocks):
        out = dense_block(out, num_layers=num_layers_per_block, growth_rate=growth_rate, name=f"dense_block_{i + 1}")
        out = transition_block(out, compression=compression, name=f"transition_{i + 1}")

    out = dense_block(out, num_layers=num_layers_per_block, growth_rate=growth_rate, name="last_dense_block")
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(256, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(128, activation='relu')(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="DenseNet")



"""
Dense Block. The most important Block in Dense net
"""
@gin.configurable
def dense_block(inputs, num_layers, growth_rate, name="dense_block"):
    """
    Implements a Dense Block for DenseNet.
    growth_rate (int): Number of filters added by each layer.
    """
    # list of feature to restore all the output
    features = [inputs]
    with tf.name_scope(name):
        for i in range(num_layers):
            out = tf.keras.layers.BatchNormalization()(features[-1])
            if name == "last_dense_block" and i == num_layers - 1:
                out = tf.keras.layers.Conv2D(growth_rate, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001), name='last')(out)
            else:
                out = tf.keras.layers.Conv2D(growth_rate, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(out)
            features.append(out)
            inputs = tf.keras.layers.Concatenate()(features)  # Concatenate input with output of the layer
    return inputs



"""
Transition layer to connect different dense layer in Net
"""
def transition_block(inputs, compression=0.5, name="transition"):
    """
    Parameters:
        compression (float): Compression factor to reduce the number of feature maps.
        name (str): Name for the layer.

    """
    num_filters = int(inputs.shape[-1] * compression)  # Calculate reduced number of filters
    with tf.name_scope(name):
        # Batch Normalization
        out = tf.keras.layers.BatchNormalization()(inputs)
        # ReLU Activation
        out = tf.keras.layers.ReLU()(out)
        # 1x1 Convolution to reduce feature map size
        out = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(out)
        # 2x2 Average Pooling for down-sampling
        out = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(out)
    return out




"""
Bottleneck Block to decrease the input channel
Bottleneck layer
{ inputs -> 1X1 -> BN -> Relu -> 3X3 -> BN -> MP -> Relu -> 1X1 -> BN + (inputs) -> 1X1 } -> Relu
"""
@gin.configurable
def bottleneck_block(inputs, filters, kernel_size, strides=(1, 1)):

    residual = inputs

    out = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1),
                                 padding='same',
                                 activation=tf.nn.leaky_relu,
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Conv2D(filters, kernel_size,
                                 padding='same',
                                 activation=tf.nn.leaky_relu,
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Conv2D(filters,  kernel_size=(1, 1),
                                 padding='same',
                                 activation=tf.nn.leaky_relu,
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(out)
    out = tf.keras.layers.BatchNormalization()(out)

    if residual.shape[-1] != out.shape[-1]:
        residual = tf.keras.layers.Conv2D(out.shape[-1], (1, 1), strides=strides, padding='same')(residual)
        residual = tf.keras.layers.MaxPool2D((2, 2))(residual)
        print("out shape:", residual.shape)

    out = tf.keras.layers.Add()([out, residual])
    return out
