import zipfile
import os
import logging
import glob
import matplotlib.pyplot as plt
import numpy as np
from data_handling import hapt_Experiments
import random
import gin
import pickle
import data_handling.tfr as tfr
from data_handling.data_class import Data
import tensorflow as tf

Data = Data()
@gin.configurable()
def import_data_from_raw_files(run_paths, window_size=250, window_shift_ratio=0.5, label_threshold=0.5,
                               SELECTED_CLASSES=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], tfr_filepath=""):
    #SELECTED_CLASSES=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # return a list of the file path with acc file list and gyro file list
    files_acc, files_gyro, label_path = read_file(run_paths)

    # Extract and preprocess the Datas and labels
    if not os.path.exists(run_paths['path_data'] + '/data_after_extraction.txt'):
        logging.info('no extracted data found, extracting data ...')
        Data.raw_data_norm = data_extraction(run_paths, files_acc, files_gyro)
        Data.label_guide = label_extraction(label_path)

    generate_label(Data.raw_data_norm, Data.label_guide)
    logging.info("label generate finished")
    Data.slide_window(window_size=250, window_shift_ratio=0.5, label_threshold=0.6)
    logging.info("windowed label and data generation finished")

    # creat the dataset and generate the TFrecord File
    # here should preprocess the dataset, combined the dataset with exp id or user id
    train_dataset = Data.packing2_dataset(user_start_id = 1, user_end_id = 21)
    test_dataset = Data.packing2_dataset(user_start_id = 22, user_end_id = 27)
    validation_dataset = Data.packing2_dataset(user_start_id = 28, user_end_id = 30)

    # Then we can deal with the dataset
    # for example we need to remove the label
    train_ds = filter_and_prepare_rnn_dataset(train_dataset, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], oversample = True)
    val_ds = filter_and_prepare_rnn_dataset(validation_dataset, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], oversample = False)
    test_ds = filter_and_prepare_rnn_dataset(test_dataset, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], oversample = False)

    return train_ds, val_ds, test_ds



# check the file and read out the dataset path
def read_file(run_paths):
    """
    :param run_paths:
    :return: files_acc, files_gyro, label_path
    """
    # datapath = os.path.join(run_paths['home'], 'HAPT_dataset/RawData/')
    datapath = os.path.join(run_paths['path_data'], 'HAPT_dataset/RawData/')
    datapath = datapath if os.path.exists(datapath) else ''
    if os.path.isdir(datapath):
        logging.info("Data set already extracted...")
    else:
        # if there is the zip file then extract the zip file
        logging.info("No data be found unter the Folder, please unzip the dataset")

    # return the two list of the
    files_acc = sorted(glob.glob(datapath + "acc_exp*.txt"))
    files_gyro = sorted(glob.glob(datapath + "gyro_exp*.txt"))
    label_path = glob.glob(datapath + "labels.txt")
    return files_acc, files_gyro, label_path


# Extract Data from files according to the data name and append to a list
def data_extraction(run_paths, files_acc, files_gyro):
    """
    :param run_paths:
    :param files_acc:
    :param files_gyro:
    :param window_size:
    :param window_shift_ratio:
    :param labeling_threshold:
    :return: combined and normalized Rawdata
    """
    # add the folder path together
    folder = os.path.join(run_paths['path_data'], 'HAPT_dataset/RawData/')

    for elist_counter, f in enumerate(files_acc):
        if all(a in f for a in ['exp', 'user', 'acc']):
            # extract the filename
            file_name = os.path.basename(f)
            # extract experiment id
            experiment = file_name.split('exp')[1].split('_user')[0]
            # extract experiment id
            user = file_name.split('_user')[1].split('.txt')[0]
            logging.info(f'Preprocessing experiment {experiment} ...')

    # THIS Function will return the normalized Raw Data
    raw_data_norm = combined_data(files_acc, files_gyro)
    return raw_data_norm


def combined_data(files_acc, files_gyro):
    """
    :param files_acc:
    :param files_gyro:
    :return: Raw_data_norm (X | Y | Z | Pitch | Yaw | Row) with Z-Score Normalization
    """
    # check if the data is equal
    assert len(files_acc) == len(files_gyro), "data broken acc unequal with gyro, check the dataset"
    # create the Rawdata list to store the data

    Raw_data_norm = []
    for list_head_acc, list_head_gyro in zip(files_acc, files_gyro):
        Raw_data = []
        temp_data_acc = []
        # temp_data_combined store the temp combined data
        with open(list_head_acc, 'r', encoding='utf-8') as file:
            for line in file:
                # change str to double then append to list
                double_array = [float(num) for num in line.split()]
                temp_data_acc.append(double_array)

        temp_data_gyro = []
        with open(list_head_gyro, 'r', encoding='utf-8') as file:
            for line in file:
                double_array = [float(num) for num in line.split()]
                temp_data_gyro.append(double_array)

        Raw_data = np.hstack((temp_data_acc, temp_data_gyro))
        Raw_data_norm.append(normalization(Raw_data))

    print("Finish the Raw data normalization")
    return Raw_data_norm


# Function to normalization the data
def normalization(data):
    """
    :param data:
    :return: normalized data
    """
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    # avoid the std value to 0
    return (data - mean_vals) / np.where(std_vals == 0, 1, std_vals)


# extract the label from the label file
def label_extraction(label_path):
    """
    :param label_path:
    :return: label guide
    """
    label_list = []
    for path in label_path:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                # change str to double then append to list
                label = [int(num) for num in line.split()]
                label_list.append(label)
    return label_list

# generate the label
def generate_label(raw_data_norm, label_guided):
    """
    geenrate the label
    :param raw_data_norm:
    :param label_guided:
    :return:
    """
    for i, row in enumerate(raw_data_norm):  # i col number and experiment id, row jetzige Col
        # every experiment length
        element_length = len(row)
        # temp_label
        temp_label = [0] * element_length
        for label_line in label_guided:
            # go through the whole label_guid
            if label_line[0] == i+1:
                start_position = label_line[3]
                end_position = label_line[4]
                temp_label[start_position:end_position + 1] = [label_line[2]] * (end_position - start_position + 1)

        Data.label.append(temp_label)

# handling the dataset and processing the dataset
def filter_and_prepare_rnn_dataset(dataset, target_remove_labels, oversample):
    # target_remove_labels = [0,1,1,1,1,1,1,1,1,1,1,1,1]  #remove class '0'
    # but I think it should be [1,0,0,0,0,0,0,0,0,0,0,0]
    """
    Filters and prepares the flattened dataset for RNN training by removing unwanted labels and corresponding data.
    Args:
        dataset (list): Flattened list of all windowed data.
        dataset (list): Flattened list of all windowed labels.
        target_remove_labels (list): A list or array indicating labels to remove.
            - 0 or False means the label is to be removed.
            - 1 or True means the label is retained.

    Returns:
        tuple: (filtered_data, filtered_labels)
            - filtered_data: Data samples after filtering.
            - filtered_labels: Corresponding labels after filtering.
    """
    # Ensure target_remove_labels is a boolean array
    target_remove_labels = np.array(target_remove_labels)
    data = np.array(dataset[0])
    label = np.array(dataset[1])

    # Create a mask for labels to keep
    mask = ~np.all(label == target_remove_labels, axis=1)

    # Apply the mask to filter data and labels
    filtered_data = data[mask]
    filtered_labels = label[mask]
    #delete the first col
    filtered_labels = np.delete(filtered_labels, 0, axis=1)

    if oversample == True:
        filtered_data,filtered_labels = balance_dataset(filtered_data,filtered_labels)

    data_tensor = tf.constant(filtered_data, dtype=tf.float32)  # Adjust dtype as needed
    labels_tensor = tf.constant(filtered_labels, dtype=tf.float32)  # Adjust dtype as needed
    dataset = tf.data.Dataset.from_tensor_slices((data_tensor, labels_tensor))
    dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size=32)
    return dataset

# balacned the dataset and oversample the data
def balance_dataset(data, label):
    # Unpack the dataset
    x = data
    y = label
    sum_all = sum(y)
    target_sample_amount = sum_all.max()

    for i in range(len(sum_all)):
        x_values = np.zeros((sum_all[i], x.shape[1], x.shape[2]))
        y_values = np.zeros((sum_all[i], y.shape[1]))
        l = 0
        for k in range(len(x)):
            if y[k][i] == 1:
                x_values[l] = x[k]
                y_values[l] = y[k]
                l += 1

        added_samples = 0
        if i == 0:
            x_values_out = x_values
            y_values_out = y_values
        else:
            x_values_out = np.concatenate((x_values_out, x_values))
            y_values_out = np.concatenate((y_values_out, y_values))
        while (sum_all[i] + added_samples) < target_sample_amount:
            # -> samples needed
            x_values_out = np.concatenate((x_values_out, np.array(random.sample(list(x_values), 1))))
            y_values_out = np.concatenate((y_values_out, y_values[0:1]))
            added_samples += 1

    return np.array(x_values_out), np.array(y_values_out)
