import tensorflow as tf
import os
import numpy as np
import logging
import pandas as pd

from input_pipeline.datasets import label_image_path, load_image

"""
The first step: change images and labels to the TFRecord Datas
The 1.5 may be augment the dataset
The second step: read the TFRecord data as the input pipeline
"""


def image_example(image_string, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))



"""
write the tfrecord data from images and labels
"""
def write_tfrecord(image_dir, label_path, output_path):
    """
    Write images and labels into a TFRecord file.

    Args:
        image_dir (str): Directory containing the images.
        csv_path (str): Path to CSV file with filenames and labels.
        output_path (str): Output path for TFRecord file.
    """
    # Read labels from the CSV file
    df = pd.read_csv(label_path)

    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in df.iterrows():
            image_path = os.path.join(image_dir, row['Image name']+'.jpg')
            label = int(row['Retinopathy grade'])  # Assuming label is an integer
            # Read and encode the image as JPEG
            image = tf.io.read_file(image_path)
            # was here should be some image processing?
            image_example_proto = image_example(image.numpy(), label)  # why here should be .numpy format?

            writer.write(image_example_proto.SerializeToString())

    print(f'TFRecord saved at {output_path}')


def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)

    # binary data decode to image
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [256, 256])  # 调整大小
    label = example['label']
    return image, label

# load the datasets
def load_dataset(tfrecord_path, batch_size):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(parse_tfrecord_fn)
    #dataset = tf.data.Dataset.from_tensor_slices((dataset))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)   # if use prefetch than tensorflow can read the dataset?
    #why?
    #dataset = dataset.repeat(-1)
    return dataset


# use dataset
# dataset = load_dataset("output.tfrecord", batch_size=32)