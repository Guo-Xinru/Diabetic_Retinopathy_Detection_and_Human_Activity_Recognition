import numpy as np
import logging
import tensorflow as tf
import gin
from data_handling import tfr, hapt_data
import wandb

import os
import matplotlib.pyplot as plt

activity_HAPT = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4: 'SITTING', 5: 'STANDING',
                 6: 'LAYING', 7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
                 11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND'}

def get_datasets(FLAGS, run_paths, sensor_pos):
    # define the TFRecord
    path_train = os.path.join(run_paths["path_data"], "train_ds.tfrecord")
    path_test = os.path.join(run_paths["path_data"], "test_ds.tfrecord")
    path_val = os.path.join(run_paths["path_data"], "val_ds.tfrecord")

    if not os.path.exists(os.path.join(run_paths["path_data"], "val_ds.tfrecord")):
        if FLAGS.source_data == "HAPT":
            logging.info("creating new TFR-Files, it will may take several mins")
            train_ds, val_ds, test_ds = hapt_data.import_data_from_raw_files(run_paths, tfr_filepath=run_paths["path_data_tfrecord"])
            # write to TFReocrd File
            write_TFrecord(train_ds,path_train)
            write_TFrecord(test_ds,path_test)
            write_TFrecord(val_ds,path_val)
            plot_random_window(train_ds)

    else:
        logging.info("Corresponding tfrecord files exist, reading from existing TFR-Files")
        train_ds = load_tfrecord_file(path_train, 32)
        test_ds = load_tfrecord_file(path_test, 32)
        val_ds = load_tfrecord_file(path_val, 32)
        plot_random_window(train_ds)
        visualize_datasets_label_distribution(train_ds, test_ds, val_ds, num_classes=12)
    """
        plot_random_window(run_paths, train_dataset, save_path=run_paths['path_data_tfrecord'])
    """
    return train_ds, val_ds, test_ds


def plot_random_window(dataset):
    # plot single window
    for data, label in dataset.take(1):
        print("Data shape:", data.shape)
        print("Label shape:", label.shape)

        # Take a batch and plot
    for idx, (data, label) in enumerate(dataset.take(1)):
        data = data.numpy()  # Convert to NumPy array for visualization
        label = tf.argmax(label, axis=1).numpy()  # Assuming one-hot encoding

        # Select a random sample from the batch
        sample_index = np.random.randint(0, data.shape[0])
        data_sample = data[sample_index]  # Shape: (250, 6)
        label_sample = label[sample_index]  # Shape: (12,)

        # Create a time axis for plotting (assuming 250 time steps)
        time_steps = np.arange(250)

        # Plot Accelerometer (x, y, z)
        plt.figure(figsize=(12, 6))
        for i, axis in enumerate(['x', 'y', 'z']):
            plt.plot(time_steps, data_sample[:, i], label=f'{axis}')
        plt.xlabel('Time Steps')
        plt.ylabel('Accelerometer Value')
        plt.title(f'Accelerometer Data (x, y, z) for Sample {sample_index} with Label: {label_sample}')
        plt.legend()
        plt.show()

        # Plot Gyroscope (pitch, yaw, roll)
        plt.figure(figsize=(12, 6))
        for i, axis in enumerate(['pitch', 'yaw', 'roll'], start=3):
            plt.plot(time_steps, data_sample[:, i], label=f'{axis}')
        plt.xlabel('Time Steps')
        plt.ylabel('Gyroscope Value')
        plt.title(f'Gyroscope Data (pitch, yaw, roll) for Sample {sample_index} with Label: {label_sample}')
        plt.legend()
        plt.show()


def calculate_label_distribution(dataset, num_classes=12):
    """
    Calculates the label distribution for a given dataset.

    Args:
        dataset: A tf.data.Dataset containing data and one-hot encoded labels.
        num_classes: Number of classes in the labels.

    Returns:
        A numpy array of label counts for each class.
    """
    label_counts = np.zeros(num_classes, dtype=int)

    for _, labels in dataset:
        labels = tf.argmax(labels, axis=1).numpy()  # Convert one-hot encoding to class indices
        for label in labels:
            label_counts[label] += 1

    return label_counts


def visualize_datasets_label_distribution(train_dataset, test_dataset, val_dataset, num_classes=12):
    """
    Visualizes the label distribution of train, test, and validation datasets.

    Args:
        train_dataset: Training dataset.
        test_dataset: Testing dataset.
        val_dataset: Validation dataset.
        num_classes: Number of classes in the labels.
    """
    # Calculate label distributions
    train_label_counts = calculate_label_distribution(train_dataset, num_classes)
    test_label_counts = calculate_label_distribution(test_dataset, num_classes)
    val_label_counts = calculate_label_distribution(val_dataset, num_classes)

    # Plot distributions
    x = np.arange(num_classes)  # Class indices
    width = 0.25  # Bar width

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, train_label_counts, width, label='Train', color='skyblue', edgecolor='black')
    plt.bar(x, test_label_counts, width, label='Test', color='lightgreen', edgecolor='black')
    plt.bar(x + width, val_label_counts, width, label='Validation', color='salmon', edgecolor='black')

    # Labeling and formatting
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Label Distribution Across Train, Test, and Validation Datasets')
    plt.xticks(x, [f'Class {i}' for i in range(num_classes)], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Print counts for further inspection
    #print("Train Label Counts:", train_label_counts)
    #print("Test Label Counts:", test_label_counts)
    #print("Validation Label Counts:", val_label_counts)


def reverse_one_hot_coding(tensor):
    ndim = tf.rank(tensor)
    tensor = tf.argmax(tensor, axis=ndim - 1)
    return tensor


"""
This four functions are the read and write TFRecord File FUnction
Write:
write_TFrecord
serialize_example

Read:
load_tfrecord_file
parse_example

"""
def write_TFrecord(dataset, filename):
    """
    Writes a dataset to a TFRecord file.

    Args:
        dataset: A tf.data.Dataset object (data and labels).
        filename: Path to the TFRecord file.
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for data, label in dataset:
            for i in range(len(data)):  # Write each sample in the batch
                serialized_example = serialize_example(data[i].numpy(), label[i].numpy())
                writer.write(serialized_example)


def serialize_example(data, label):
    """
    Serializes a single example for TFRecord format.

    Args:
        data: Sensor data (numpy array, shape: [250, 6]).
        label: One-hot encoded label (numpy array, shape: [12]).

    Returns:
        Serialized TFRecord example.
    """
    feature = {
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten())),  # Flatten the data
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def parse_example(serialized_example):
    """
    Parses a single TFRecord example to data and label.

    Args:
        serialized_example: A serialized TFRecord example.

    Returns:
        data: Parsed sensor data (shape: [250, 6]).
        label: Parsed label (shape: [12]).
    """
    feature_description = {
        'data': tf.io.FixedLenFeature([250 * 6], tf.float32),  # 250 time steps x 6 sensors
        'label': tf.io.FixedLenFeature([12], tf.float32)  # One-hot encoded label (12 classes)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    data = tf.reshape(example['data'], [250, 6])  # Reshape data to original shape
    label = example['label']
    return data, label


def load_tfrecord_file(filename, batch_size=32):
    """
    Loads a TFRecord file into a tf.data.Dataset.

    Args:
        filename: Path to the TFRecord file.
        batch_size: Batch size for batching the dataset.

    Returns:
        A tf.data.Dataset object containing parsed data and labels.
    """
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(parse_example)  # Parse each example

    # Batch the dataset and prefetch for performance
    batched_dataset = parsed_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return batched_dataset
