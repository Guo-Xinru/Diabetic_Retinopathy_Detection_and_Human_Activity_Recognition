import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



def create_cnn_model(input_shape=(256, 256, 3), num_classes=10):
    model = tf.keras.Sequential([

        # 第一层卷积块
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 第二层卷积块
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 第三层卷积块
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 第四层卷积块
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),

        # 全连接层
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # 防止过拟合

        # 输出层
        tf.keras.layers.Dense(num_classes, activation='softmax')  # 分类任务使用softmax，回归可以用linear
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # 若为回归任务，可用'mean_squared_error'
                  metrics=['accuracy'])
    return model


def train_model(dataset, model, batch_size, epochs):
    history = (model.fit(dataset, batch_size=batch_size, epochs=epochs))
    return history, model



def plot_training_history(history):
    """
    plot the loss and accuracy
    """
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 绘制准确度曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()

