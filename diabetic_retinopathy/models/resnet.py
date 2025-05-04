import tensorflow as tf
import gin

@gin.configurable
def res_net(input_shape, n_classes, n_blocks, filters, block_type="basic"):
    """
    Constructs a ResNet model.
    Created from chatGPT
        input_shape (tuple): Shape of the input tensor, e.g., (224, 224, 3).
        n_classes (int): Number of output classes.
        n_blocks (int): Number of residual blocks per stage.
        filters (list): List of filters for each stage.
        block_type (str): Type of block, either "basic" or "bottleneck".
    """
    assert len(filters) == len(n_blocks), "Length of filters must match the number of blocks."

    inputs = tf.keras.Input(input_shape)
    # Initial Conv + BN + ReLU + MaxPooling
    out = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(out)

    # Residual Blocks
    for i, num_blocks in enumerate(n_blocks):
        for j in range(num_blocks):
            strides = (1, 1)
            if j == 0 and i > 0:  # Apply stride of 2 for downsampling at the start of each stage (except the first)
                strides = (2, 2)
            if block_type == "bottleneck":
                out = bottleneck_block(out, filters[i], strides=strides, name=f"bottleneck_block_{i + 1}_{j + 1}")
            else:
                out = basic_block(out, filters[i], strides=strides, name=f"basic_block_{i + 1}_{j + 1}")

    # Global Average Pooling and Dense Output
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(out)
    #outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet")


def basic_block(inputs, filters, strides=(1, 1), name="basic_block"):
    """
    Implements a Basic Residual Block for ResNet-18/34.

    Parameters:
        inputs (tf.Tensor): Input tensor.
        filters (int): Number of filters for the convolutional layers.
        strides (tuple): Strides for the first convolution.
        name (str): Block name.

    Returns:
        tf.Tensor: Output tensor of the basic block.
    """
    with tf.name_scope(name):
        residual = inputs

        # First Conv-BN-ReLU
        out = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)

        # Second Conv-BN
        out = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     kernel_regularizer=tf.keras.regularizers.L2(0.01))(out)
        out = tf.keras.layers.BatchNormalization()(out)

        # Adjust residual to match dimensions if necessary
        if residual.shape[-1] != out.shape[-1] or strides != (1, 1):
            residual = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(residual)

        # Add residual and apply activation
        out = tf.keras.layers.Add()([out, residual])
        out = tf.keras.layers.ReLU()(out)
        return out


def bottleneck_block(inputs, filters, strides=(1, 1), name="bottleneck_block"):
    """
    Implements a Bottleneck Residual Block for ResNet-50/101/152.

    Parameters:
        inputs (tf.Tensor): Input tensor.
        filters (int): Number of filters for the bottleneck layers.
        strides (tuple): Strides for the first convolution.
        name (str): Block name.

    Returns:
        tf.Tensor: Output tensor of the bottleneck block.
    """
    with tf.name_scope(name):
        residual = inputs

        # First Conv-BN-ReLU (1x1)
        out = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)

        # Second Conv-BN-ReLU (3x3)
        out = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                     kernel_regularizer=tf.keras.regularizers.L2(0.01))(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)

        # Third Conv-BN (1x1)
        out = tf.keras.layers.Conv2D(filters * 4, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                     kernel_regularizer=tf.keras.regularizers.L2(0.01))(out)
        out = tf.keras.layers.BatchNormalization()(out)

        # Adjust residual to match dimensions if necessary
        if residual.shape[-1] != out.shape[-1] or strides != (1, 1):
            residual = tf.keras.layers.Conv2D(filters * 4, kernel_size=(1, 1), strides=strides, padding='same')(
                residual)

        # Add residual and apply activation
        out = tf.keras.layers.Add()([out, residual])
        out = tf.keras.layers.ReLU()(out)
        return out
