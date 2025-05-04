import gin
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import cv2
from tensorflow.python.ops.numpy_ops import np_config
import os


"""
deep visualization using Grad-CAM,
use the RELU function to calculate the heat map Gram CAM
use the Guided back propergation to get the Guided Gram CAM map
add the both map to get the Grad-CAM
"""
class Visualisation:
    def __init__(self, image_path, save_path, model, input_shape):
        # height width channel from a image
        self.h, self.w, self.c = input_shape
        self.path = save_path

        image = cv2.imread(image_path)
        self.image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
        self.image = tf.expand_dims(tf.cast(self.image, tf.float32), 0)
        self.grey_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        self.model = model

        # get the last convolution layer and
        # creat a model which use the input the get last conv layer output und model output
        # last_conv_layer = self.model.get_layer()
        self.last_conv_layer = None
        for layer in reversed(self.model.layers):
            if layer.name == 'last' or 'coder' in layer.name:
                self.last_conv_layer = layer
                break
        self.last_conv_layer_model = tf.keras.Model(model.inputs, [self.last_conv_layer.output, model.output])


    """
    Display the image 
    """
    def MapShow(self, grey_image, heatmap, save_path, title, name):
        np_config.enable_numpy_behavior()
        heatmap = 0.8 * heatmap + grey_image
        heatmap = (heatmap / np.max(heatmap) * 255).astype('uint8')
        plt.matshow(heatmap)
        plt.title('{}'.format(title))
        plt.savefig(f'{save_path}/{title+name}.png')


    """
    GradCAM - Process
    """
    def GradCAM(self, name):
        with tf.GradientTape() as tape:
            feature_map, output = self.last_conv_layer_model(self.image)
            index = tf.argmax(output, axis=-1)[0]
            output = output[:, index]
        gradient = tape.gradient(output, feature_map)
        grad_average = tf.reduce_mean(gradient, axis=(0,1,2))

        # change feature map to the heatmap
        feature_map = feature_map.numpy()[0]
        for i in range(feature_map.shape[-1]):
            feature_map[:, :, i] *= grad_average.numpy()[i]
        heatmap = np.mean(feature_map, axis=-1)

        # heatmap generation
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = cv2.resize(heatmap, (self.h, self.w), cv2.INTER_LINEAR)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        self.MapShow(self.grey_image, heatmap, self.path, 'Grad-CAM', name)
        return heatmap / 255.0


    def GuidedBackpropagation(self, name):
        with tf.GradientTape() as tape:
            tape.watch(self.image)
            output = self.model(self.image)
            index = tf.argmax(output, axis=-1)[0]
            output = output[:, index]
        gbp_map = tape.gradient(output, self.image)[0].numpy()

        gbp_map = cv2.resize(gbp_map, (self.h, self.w), cv2.INTER_LINEAR)
        np_config.enable_numpy_behavior()
        gbp_map = (gbp_map - gbp_map.mean()) * 0.25 / (gbp_map.std() + tf.keras.backend.epsilon())
        gbp_map = (np.clip(gbp_map + 0.5, 0.25, 1) * 255).astype('uint8')
        gbp_map = cv2.cvtColor(gbp_map, cv2.COLOR_BGR2RGB)

        self.MapShow(self.grey_image, gbp_map, self.path, 'Guided Back-Propagation', name)
        return gbp_map / 255.0


    def guidedGradCAM(self, name):

        """ define guided grad-CAM process """

        ggcam_map = self.GuidedBackpropagation(name) * self.GradCAM(name)
        ggcam_map = np.uint8(255 * ggcam_map)
        self.MapShow(self.grey_image, ggcam_map, self.path, 'Guided Grad_CAM', name)
