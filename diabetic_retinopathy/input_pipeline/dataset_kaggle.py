import os.path
import glob
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import cv2
import numpy as np
import os


def load_kaggle(name, data_dir):
    """
    :param name:
    :param data_dir:
    :return:
    """
    logging.info(f"Preparing dataset {name}...")
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'diabetic_retinopathy_detection/btgraham-300',
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        with_info=True,
        data_dir=data_dir
    )

    def _preprocess(img_label_dict):
        return img_label_dict['image'], img_label_dict['label']

    ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return prepare(ds_train, ds_val, ds_test, ds_info)

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info


def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.
    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    return image, label


def augment(image, label):
    """Data augmentation"""
    image = tf.image.resize(image, [300, 300])  # resize the iamge to 36ß
    image = tf.image.random_crop(image, [256, 256, 3]) # random crop
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    return image, label

