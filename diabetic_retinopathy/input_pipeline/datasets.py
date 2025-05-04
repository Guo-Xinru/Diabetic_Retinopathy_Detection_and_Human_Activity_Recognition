import cv2
import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


AUTOTUNE = tf.data.experimental.AUTOTUNE

# read the label under the dataset
# the dataset should be the same folder with the py code
def label_image_path(images_dir,label_dir, load_mode):
    """
    Args:
        images_dir (str): image path
        label_path (str): label path

    Returns:
        file_paths (list): image path list
        labels (np.ndarray): label list
     """
    # check it the label file exist
    try:
        check = pd.read_csv(label_dir)
    except FileNotFoundError:  # if no label file , exit the main function
        print("not csv label found")
        exit(1)

    label = pd.read_csv(label_dir)  # read all the label as panda format
    data = np.array(label.iloc[:, :3])  # read only the front 3 col of labels
    file_paths = [os.path.join(images_dir, fname+'.jpg') for fname in label['Image name']]
    labels = label['Retinopathy grade'].values

    if '2' in load_mode:
        # binary classification
        label_list = labels.tolist()
        binar_label = []
        for i in label_list:
            if i == 1 or i == 0:
                binar_label.append(0)
            elif i==2 or i==3 or i==4:
                binar_label.append(1)
        print(label_list)
        print(binar_label)
        binarnplabels = np.array(binar_label)
        return file_paths, binarnplabels
    else:
        return file_paths, labels


def load(name, data_dir, label_dir, load_mode, balance_classes, transfer_learning):
    if name == "idridtrain":
        logging.info(f"Preparing idridtrain {name}...")
        # load the image and zip with the label
        image_path, labels = label_image_path(data_dir, label_dir, load_mode)
        dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
        print('datapath load finish, now loading image')
        dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset).numpy() // 10) # the site of dataset / 10
        dataset = dataset.map(load_image)

        # test code
        # analyze_dataset(labels)
        if balance_classes:
            dataset = balance_dataset(dataset)
        label_counts = get_label_counts(dataset)
        print("Label counts:", label_counts)

        if "transformer" in load_mode:
            dataset = dataset.map(augment_transformer)
            dataset = dataset.repeat(3)  # repeat the dataset
        else:
            dataset = dataset.map(augment)
            if transfer_learning:
                dataset = dataset.repeat(3)
            else:
                dataset = dataset.repeat(-1)  # repeat the dataset
            # if vision transformer then dont batch it
            dataset = dataset.batch(batch_size=32)  # batch the dataset the ensure the input shape
        return dataset

    elif name == "idridtest":
        logging.info(f"Preparing idridtrain {name}...")
        # load the image and zip with the label
        image_path, labels = label_image_path(data_dir, label_dir, load_mode)
        dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
        print('datapath test load finish, now loading image')
        if "transformer" in load_mode:
            dataset = dataset.map(load_test_transformer_image)
            label_counts = get_label_counts(dataset)
            print("Label counts:", label_counts)
            dataset = dataset.repeat(3)  # repeat the dataset
        else:
            dataset = dataset.map(load_test_image)
            label_counts = get_label_counts(dataset)
            print("Label counts:", label_counts)
            dataset = dataset.repeat(3)  # repeat the dataset
            # if vision transformer then dont batch it
            dataset = dataset.batch(batch_size=32)  # batch the dataset the ensure the input shape
        return dataset
    else:
        raise ValueError


"""
load iamge !
"""
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])  # change the size of image but i think it is unnecessary#
    image = tf.cast(image, tf.float32) # / 255.0
    save_load_train_path = os.getcwd() + '/visusalization/loadtrainimage.jpg'
    # save_and_display_image(image, file_path="/home/kusabi/DLLAB/loadtrainimage.jpg")
    return image, label

"""
load test iamge !
"""
def load_test_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])  # change the size of image but i think it is unnecessary#
    image = tf.cast(image, tf.float32) #/ 255.0
    save_load_test_path = os.getcwd() + '/visusalization/loadtestimage.jpg'
    # save_and_display_image(image, file_path="/home/kusabi/DLLAB/loadtestimage.jpg")
    return image, label


"""
load transformer test image !
"""
def load_test_transformer_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])  # change the size of image but i think it is unnecessary#
    image = tf.cast(image, tf.float32) #/ 255.0
    logging.info(f"Preparing test image transformer...")
    image = tf.image.resize(image, [256, 256])
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label



"""
# Data Augment: 
1. Augment images and save it locally (small dataset idrid suitable)
2. Augment images when load to the dataset (big dataset maybe Kaggle suitable)
"""
# this rotate layer should be created outside of the function
# otherwise it will return the tf.variable call mehr times error
random_rotation_layer = tf.keras.layers.RandomRotation(factor=0.05)  # create random rotate layers
seed = np.random.seed(525)
def augment(image, label):
    """Data augmentation"""
    # random_angles = tf.random.Generator.from_seed(shape=(),)
    # random_angles = tf.random.uniform(shape=(), minval=-np.pi / 12, maxval=np.pi / 12, )
    image = random_rotation_layer(image)  # random rotate
    image = tf.image.resize(image, [300, 300])  # resize the iamge to 36ß
    image = tf.image.random_crop(image, [256, 256, 3]) # random crop
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, 0.15)  # random brightness
    # image = tf.cast(image, tf.float32) # normalization ? should it 100%?
    # save_and_display_image(image, file_path="/home/kusabi/DLLAB/augmented_image.jpg")
    return image, label


"""
# Data Augment for vision transformer
1. Augment images and save it locally (small dataset idrid suitable)
2. Augment images when load to the dataset (big dataset maybe Kaggle suitable)
3. need to change the channel of the transformer
"""
def augment_transformer(image, label):
    """Data augmentation"""
    image = random_rotation_layer(image)  # random rotate
    image = tf.image.resize(image, [300, 300])  # resize the iamge to 36ß
    image = tf.image.random_crop(image, [256, 256, 3]) # random crop
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    print(image.shape)
    logging.info(f"Preparing transformer image ...")
    image = tf.image.resize(image, [256, 256])
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label


def save_and_display_image(image, file_path="/home/kusabi/DLLAB/augmented_image.jpg"):
    """Save image to disk and display it"""
    # Cast image to uint8 and scale to [0, 255]
    # image = tf.cast(image * 255.0, tf.uint8)
    image = tf.cast(image * 1.0, tf.uint8)
    # Save the image
    tf.io.write_file(file_path, tf.image.encode_jpeg(image))



"""
load test data set need to change in future
ds_val, ds_test is different with ds_train
for ds_train we have prepare for the repeat(-1) but val and test ds shouldnt have repeat
def load_test(data_dir, label_dir):
    logging.info(f"Preparing test dataset")
    # load the image and zip with the label
    image_path, labels = label_image_path(data_dir, label_dir)
    dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
    print('datapath load finish, now loading image')

    dataset = dataset.map(load_test_image)
    label_counts = get_label_counts(dataset)
    print("Label counts:", label_counts)
    # dataset = dataset.repeat(1)  # repeat the dataset
    for images, labels in dataset.take(1):
        print("Images shape:", images.shape)  # shoulde be (32, 256, 256, 3)
        print("Labels shape:", labels.shape)  # shoulde be  (32,)
    dataset = dataset.batch(batch_size=32)  # batch the dataset the ensure the input shape
    return dataset
"""

"""
Analyse the dataset, only 
"""
def analyze_dataset(labels):

    unique_elements, counts = np.unique(labels, return_counts=True)
    # out put every labels occur time in dataset
    for element, count in zip(unique_elements, counts):
        print(f"label {element} is {count}")

    plt.bar(unique_elements, counts)

    total_count = len(labels)
    for i, (count, element) in enumerate(zip(counts, unique_elements)):
        percentage = count / total_count * 100
        plt.text(i, count + 0.5, f'{count} ({percentage:.2f}%)', ha='center', va='bottom', fontsize=10)
    # add title , x and y axis
    plt.title("Dadaset distribution")
    plt.xlabel("Labels")
    plt.ylabel("Nummber Labels")
    # show the diagramm
    save_path = os.getcwd() + '/visusalization/dataset.jpg'
    plt.savefig(save_path)
    plt.show()


"""
Balance the dataset with downsample or Upsample
"""
import tensorflow as tf
import numpy as np

def balance_dataset(dataset, target_size=None, seed=42):
    # Step 1: Group by labels
    grouped_data = {}
    for image, label in dataset.as_numpy_iterator():
        label = int(label)  # Ensure label is int
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(image)

    # Step 2: Determine target size for each label
    min_count = min(len(images) for images in grouped_data.values())
    max_count = max(len(images) for images in grouped_data.values())
    if target_size is None:
        target_size = max_count  # Default to the smallest class size

    # Step 3: Downsample or oversample to match target size
    balanced_data = []
    rng = np.random.default_rng(seed)  # For reproducibility
    for label, images in grouped_data.items():
        if len(images) > target_size:  # Downsample
            sampled_images = rng.choice(images, target_size, replace=False)
        else:  # Oversample
            sampled_images = rng.choice(images, target_size, replace=True)
        balanced_data.extend((img, label) for img in sampled_images)

    # Step 4: Shuffle and convert back to TensorFlow dataset
    rng.shuffle(balanced_data)
    images, labels = zip(*balanced_data)
    # analyze_dataset(labels)
    balanced_dataset = tf.data.Dataset.from_tensor_slices((list(images), list(labels)))
    return balanced_dataset



"""
Count the label size in the dataset
"""
def get_label_counts(dataset):
    label_counts = Counter()
    for _, label in dataset.as_numpy_iterator():
        label_counts[int(label)] += 1
    return label_counts

