from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


def build_mobilenet_v3(input_shape=(256, 256, 3), num_classes=2, base_model_name='name'):
    """
    build the transfer Learning Model
    select which base model should be loaded
    """
    if base_model_name=='mobilev3large':
        base_model = MobileNetV3Large(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
    elif base_model_name=='mobilev3small':
        base_model = MobileNetV3Small(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        for layer in base_model.layers[-3:]:
            layer.trainable = True


    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.summary()
    return model


"""
Train step for the model
"""
def train_model(ds_train, ds_val, epochs, model):
    """
    Train the model, maybe can implement the deep visu to the transfer learning
    init the params
    """
    best_val_loss = 999
    best_accuracy = 0

    # model = build_lightweight_cnn(input_shape=(256, 256, 3), num_classes=5)

    # build MobileNetV3 model
    # model = build_mobilenet_v3(input_shape=(256, 256, 3), num_classes=2,base_model_name='name')

    # learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.002,
        decay_steps=10000,
        decay_rate=0.96
    )

    # optimizers Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile Model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # History of the train loss
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'grad_norms': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # record epoch gradient
        epoch_gradient_norms = []

        # manuelle every gradient
        for step, (images, labels) in enumerate(ds_train):
            print('the' + str(step))
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                loss = tf.reduce_mean(loss)

            # calculate and apply the gradient
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # record every l2 gradient norm
            batch_gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
            epoch_gradient_norms.append(np.mean(batch_gradient_norms))

        # record it into history
        history['grad_norms'].append(np.mean(epoch_gradient_norms))

        # train and test loss and accurancy
        train_metrics = model.evaluate(ds_train, verbose=0)
        val_metrics = model.evaluate(ds_val, verbose=0)
        history['train_loss'].append(train_metrics[0])
        history['train_accuracy'].append(train_metrics[1])
        history['val_loss'].append(val_metrics[0])
        history['val_accuracy'].append(val_metrics[1])

        print(f"Train Loss: {train_metrics[0]:.4f}, Train Accuracy: {train_metrics[1]:.4f}")
        print(f"Validation Loss: {val_metrics[0]:.4f}, Validation Accuracy: {val_metrics[1]:.4f}")
        print(f"Average Gradient Norm for Epoch {epoch + 1}: {history['grad_norms'][-1]:.4f}")

        validation_loss = val_metrics[0]
        validation_accuracy = val_metrics[1]
        # save the weighted from the
        print("save the model weight")
        print(validation_loss)
        print(validation_accuracy)
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            print(best_val_loss)
            model.save_weights(os.getcwd() + '/weight/' + str(epoch) + 'best.weights.h5')

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            print(best_accuracy)
            model.save_weights(os.getcwd() + '/weight/' + str(epoch) + 'bestaccu.weights.h5')




        # print the train the validation label
        print("Checking one batch from validation set...")
        # for images, labels in ds_val.take(2):
        for images, labels in ds_val:
            predictions = model(images, training=False)
            predicted_labels = np.argmax(predictions, axis=1)
            print("True labels:  ", labels.numpy())
            print("Predicted labels:  ", predicted_labels)
            break

    plot_metrics(history)


def plot_metrics(history):
    """
        visualization, train loss, test loss

    """
    epochs_range = range(len(history['train_loss']))

    plt.figure(figsize=(18, 5))

    # 绘制训练和验证损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制训练和验证准确率
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # 绘制梯度范数曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['grad_norms'], label='Gradient Norms')
    plt.xlabel('Epochs')
    plt.ylabel('Average Gradient Norm')
    plt.title('Gradient Norms Across Epochs')
    plt.legend()

    plt.show()