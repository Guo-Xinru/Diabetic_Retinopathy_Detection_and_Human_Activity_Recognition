import gin
import logging
import cv2
import numpy as np
import os
import tensorflow as tf
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate, inference
from evaluation import deepvisu
from input_pipeline import datasets, TFRecord, dataset_kaggle
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models import resnet, densenet
from models.classiccnn import create_cnn_model, train_model, plot_training_history
from models import vision_Transformer, transfer_learning
from tensorflow.keras.applications import MobileNetV3Small

"""
# we choose the model here and also all stuffs here, like the weight save the train model or evaluation model
# all the paths should be the relativ path to fit the job on server
# Model should be choose from the folgende:
vgg2    # vgg like structrue with res block binary classification
vgg5    # vgg like structrue with res block 5 class
dense2  # cnn with dense block binary classification
dense5  # cnn with dense block 5 classfication
mobilenetv3large2 # transfer_learning with model v3large 2 classification
mobilenetv3large5 # transferlearning with moden mobilenetv3large 5 classification
mobilenetv3small2 # transferlearning with moden mobilenetv3small2 2 classification
transformer # vision transformer
"""

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('model', "mobilenetv3small2", 'Select which model will be trained')
flags.DEFINE_boolean('resume', False, 'resume the training from the Checkpoint') # default False
flags.DEFINE_boolean('visu',False,'check if we need to save the deep visualisation results')


def main(argv):

    logging.info('Running the Programm on local device')
    # get the dataset path on local
    train_image_path = os.getcwd() + '/idrid/IDRID_dataset/images/train'
    test_image_path = os.getcwd() + '/idrid/IDRID_dataset/images/test'
    train_label_dir = os.getcwd() + '/idrid/IDRID_dataset/labels/train.csv'  # change the path
    test_label_dir = os.getcwd() + '/idrid/IDRID_dataset/labels/test.csv'  # change the path

    # load_mode is depend on the output goal.
    # if 5 classes then the dataset will have 5 classes label
    # if 2 classes then the dataset will have only binary label
    # specific dataset process with vision transformer
    if '2' in FLAGS.model:
        if 'mobile' in FLAGS.model:
            train_dataset = datasets.load("idridtrain", train_image_path, train_label_dir, load_mode='2', balance_classes=True, transfer_learning=True)
        else:
            train_dataset = datasets.load("idridtrain", train_image_path, train_label_dir, load_mode='2',
                                          balance_classes=True, transfer_learning=False)
        test_dataset = datasets.load("idridtest", test_image_path, test_label_dir, load_mode='2', balance_classes=False, transfer_learning=False)

    elif '5' in FLAGS.model:
        if 'mobile' in FLAGS.model:
            train_dataset = datasets.load("idridtrain", train_image_path, train_label_dir, load_mode='5',balance_classes=True, transfer_learning=True)
        else:
            train_dataset = datasets.load("idridtrain", train_image_path, train_label_dir, load_mode='5',
                                          balance_classes=True, transfer_learning=False)
        test_dataset = datasets.load("idridtest", test_image_path, test_label_dir, load_mode='5', balance_classes=False, transfer_learning=False)

    else:
        train_dataset = datasets.load("idridtrain", train_image_path, train_label_dir, load_mode='transformer', balance_classes=True, transfer_learning=False)
        test_dataset = datasets.load("idridtest", test_image_path, test_label_dir, load_mode='transformer', balance_classes=False, transfer_learning=False)


    # check the dataset size and shape
    count = tf.data.experimental.cardinality(train_dataset).numpy() # count the size of dataset
    print("train_dataset size:", count)
    print(train_dataset.element_spec)
    for images, labels in train_dataset.take(1):
        print("Images shape:", images.shape)  # shoulde be (32, 256, 256, 3)
        print("Labels shape:", labels.shape)  # shoulde be  (32,)
    for images, labels in test_dataset.take(1):
        print("Images shape:", images.shape)  # shoulde be (32, 256, 256, 3)
        print("Labels shape:", labels.shape)  # shoulde be  (32,)

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # save path for deep visualization
    image_path = os.getcwd() + '/visusalization/test.jpg'
    save_path = os.getcwd() + '/visusalization/'

    if FLAGS.train and 'vgg' in FLAGS.model:
        if '2' in FLAGS.model:
            model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
        elif '5' in FLAGS.model:
            model = vgg_like(input_shape=(256, 256, 3), n_classes=5)

        trainer = Trainer(model, train_dataset, test_dataset, test_dataset, run_paths, resume=FLAGS.resume)
        epoch = 0
        visualizer = deepvisu.Visualisation(image_path, save_path, model, input_shape=(256, 256, 3))
        for _ in trainer.train():
            epoch += 1
            # if the visu flag is true then save visu foto every 2 epochs
            if FLAGS.visu == True and epoch % 2 == 0:
                    visualizer.GradCAM(name = str(FLAGS.model)+str(epoch))
                    visualizer.GuidedBackpropagation(name = str(FLAGS.model)+str(epoch))
                    visualizer.guidedGradCAM(name = str(FLAGS.model)+str(epoch))
            continue

    if FLAGS.train and 'dense' in FLAGS.model:
        if '2' in FLAGS.model:
            model = densenet.dense_net(input_shape=(256, 256, 3), n_classes=2)
        elif '5' in FLAGS.model:
            model = densenet.dense_net(input_shape=(256, 256, 3), n_classes=5)

        trainer = Trainer(model, train_dataset, test_dataset, test_dataset, run_paths, resume=FLAGS.resume)
        epoch = 0
        visualizer = deepvisu.Visualisation(image_path, save_path, model, input_shape=(256, 256, 3))
        for _ in trainer.train():
            epoch += 1
            # if the visu flag is true then save visu foto every 2 epochs
            if FLAGS.visu == True and epoch % 2 == 0:
                    visualizer.GradCAM(name = str(FLAGS.model)+str(epoch))
                    visualizer.GuidedBackpropagation(name = str(FLAGS.model)+str(epoch))
                    visualizer.guidedGradCAM(name = str(FLAGS.model)+str(epoch))
            continue

    if FLAGS.train and 'mobilenetv3large2' in FLAGS.model:
        model = transfer_learning.build_mobilenet_v3(input_shape=(256, 256, 3), num_classes=2,
                                                    base_model_name='mobilev3large')
        transfer_learning.train_model(train_dataset, test_dataset, epochs=30, model=model)

    if FLAGS.train and 'mobilenetv3small2' in FLAGS.model:
        model = transfer_learning.build_mobilenet_v3(input_shape=(256, 256, 3), num_classes=2,
                                                     base_model_name='mobilev3small')
        transfer_learning.train_model(train_dataset, test_dataset, epochs=30, model=model)

    if FLAGS.train and 'mobilenetv3large5' in FLAGS.model:
        model = transfer_learning.build_mobilenet_v3(input_shape=(256, 256, 3), num_classes=5,
                                                    base_model_name='mobilev3large')
        transfer_learning.train_model(train_dataset, test_dataset, epochs=30, model=model)

    if FLAGS.train and 'mobilenetv3small5' in FLAGS.model:
        model = transfer_learning.build_mobilenet_v3(input_shape=(256, 256, 3), num_classes=5,
                                                     base_model_name='mobilev3small')
        transfer_learning.train_model(train_dataset, test_dataset, epochs=30, model=model)

    if FLAGS.model == 'transformer' and FLAGS.train == True:
        #define the config parameters for Transformer
        tconf = vision_Transformer.TrainerConfig(max_epochs=10000, batch_size=32, learning_rate=1e-3)
        # early the patch size is 4
        model_config = {"image_size": 256,
                        "patch_size": 16,
                        "num_classes": 5,
                        "dim": 64,
                        "depth": 3,
                        "heads": 4,
                        "mlp_dim": 128}

        # the fourth and sixth should be the dataset lengh
        trainer = vision_Transformer.Trainer(vision_Transformer.ViT, model_config, train_dataset, 413 ,
                                             test_dataset, 103, tconf)
        trainer.train()

    # evaluate the models
    if FLAGS.train == False:
        if 'vgg2' in FLAGS.model:
            model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
            checkpoint = os.getcwd() + '/weight/vgg2.weights.h5'
        elif 'vgg5' in FLAGS.model:
            model = vgg_like(input_shape=(256, 256, 3), n_classes=5)
            checkpoint = os.getcwd() + '/weight/vgg5.weights.h5'
        elif 'dense2' in FLAGS.model:
            model = densenet.dense_net(input_shape=(256, 256, 3), n_classes=2)
            checkpoint = os.getcwd() + '/weight/dense2.weights.h5'
        elif 'dense5' in FLAGS.model:
            model = densenet.dense_net(input_shape=(256, 256, 3), n_classes=5)
            checkpoint = os.getcwd() + '/weight/dense5.weights.h5'
        elif 'mobilenetv3small2' in FLAGS.model:
            model = transfer_learning.build_mobilenet_v3(input_shape=(256, 256, 3), num_classes=2,
                                                         base_model_name='mobilev3small')
            checkpoint = os.getcwd() + '/weight/v3s.weights.h5'

        evaluate(model,checkpoint,test_dataset,test_dataset,run_paths)


if __name__ == "__main__":
    app.run(main)