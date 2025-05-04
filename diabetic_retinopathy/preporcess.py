import cv2
import numpy as np
import os
import tensorflow as tf
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('server', True, 'Propercess the data on server')

def preprocessing_filter_greenchannel(origin_image):
    scale_percent = 20  #
    width = int(origin_image.shape[1] * scale_percent / 100)
    height = int(origin_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(origin_image, dim, interpolation=cv2.INTER_LANCZOS4)

    alpha = 1.5
    beta = 0
    cv2.convertScaleAbs(resized_image,resized_image,alpha,beta)

    Gaussian_image = cv2.bilateralFilter(resized_image, 5, 15, 15)
    Gaussian_image[1, 1, :] = 0
    Gray_image = cv2.cvtColor(Gaussian_image,cv2.COLOR_RGB2GRAY)

    #cv2.imshow(" Gray_image", Gray_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    """""
    #now it is the algo from the Benji
    height, width = Gaussian_image.shape[:2]
    result_image = np.zeros_like(Gaussian_image)
    #define the kernel size
    kernel_size = 15
    for i in range(height):
        for j in range (width):
            #define the window bondary
            start_x = max(j - kernel_size // 2, 0)
            end_x = min(j + kernel_size // 2, width - 1)
            start_y = max(i - kernel_size // 2, 0)
            end_y = min(i + kernel_size // 2, height - 1)
            #calculate the local average
            local_patch = Gaussian_image[start_y:end_y, start_x:end_x]
            local_avg_color = local_patch.mean(axis=(0, 1))
            #substract the local average
            result_image[i, j] = Gaussian_image[i, j] - local_avg_color
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    # extract the RGB Channel convert the color image to gray with weight
    B, G, R = result_image[:, :, 0], result_image[:, :, 1], result_image[:, :, 2]
    # convert to gray with weight
    gray_image = 0.5 * R + 0 * G + 0.5 * B
    gray_image = gray_image.astype(np.uint8)  # 将结果转换为 uint8 类型
    inverted_image = cv2.bitwise_not(gray_image)
    """""
    return Gray_image


# perprocessing with histrgram enhanced
def histrogram_enhanced(origin_image):
    crop_image = origin_image[0:2848, 150:3800] # crop the black area only for this dataset
    letterbox_image = letterbox(crop_image)     # letterbox the make the image to Quadrat
    resized_image = cv2.resize(letterbox_image, (256,256), interpolation=cv2.INTER_LANCZOS4)  #resize the image to 224
    Gray_image = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)  # change the color to gray

    equ = cv2.equalizeHist(Gray_image)
    res = np.hstack((Gray_image, equ))  # stacking images side-by-side
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return equ


# perprocessing with Contrast Limited Adaptive Histogram Equalization
def histrogram_CLAHE(origin_image):
    crop_image = origin_image[0:2848, 150:3800] # crop the black area only for this dataset
    letterbox_image = letterbox(crop_image)     # letterbox the make the image to Quadrat
    resized_image = cv2.resize(letterbox_image, (256,256), interpolation=cv2.INTER_LANCZOS4)  #resize the image to 224
    #Gray_image = cv2.cvtColor(resized_image,cv2.COLOR_RGB2GRAY)  # change the color to gray
    # create a CLAHE object (Arguments are optional).
    #original cliplimit is 5
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))   # cliplimit bigger will increase the noise
    # cl1 = clahe.apply(Gray_image)
    #new propercessing


    image_stack = np.stack([
        clahe.apply(resized_image[..., 0]),  # 对 R 通道
        clahe.apply(resized_image[..., 1]),  # 对 G 通道
        clahe.apply(resized_image[..., 2])   # 对 B 通道
    ], axis=-1)

    for i in range(3):
        image_stack[..., i] = cv2.bilateralFilter(image_stack[..., i], d=3, sigmaColor=75, sigmaSpace=75)

    # 将均衡化后的图像重新缩放回 [0, 1] 并转换为 TensorFlow 张量
    image_stack = image_stack / 255.0
    image = tf.convert_to_tensor(image_stack, dtype=tf.float32)

    image = image.numpy() * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)

    mask = np.zeros(image.shape)     #  create a mask für original image
    mask = cv2.circle(mask, (mask.shape[0] // 2, mask.shape[1] // 2),
                      int(min(mask.shape[0], mask.shape[1]) * 0.5 * 0.95), (1, 1, 1), -1, 8, 0)
    mask = cv2.rectangle(mask, (0, 0), (mask.shape[0], 28), (0, 0, 0), -1, 8)
    mask = cv2.rectangle(mask, (0, 228), (mask.shape[0], mask.shape[1]), (0, 0, 0), -1, 8)
    image = image * mask

    return image



# Benjiamin Graham Algorithmus
def Graham(origin_image):

    crop_image = origin_image[0:2848, 150:3800] # crop the black area only for this dataset
    letterbox_image = letterbox(crop_image)     # letterbox the make the image to Quadrat
    resized_image = cv2.resize(letterbox_image, (256,256), interpolation=cv2.INTER_LANCZOS4)  #resize the image to 224

    image = cv2.addWeighted(resized_image, 3.8, cv2.GaussianBlur(resized_image, (0, 0), resized_image.shape[0] / 16), -4, 128)
    image = cv2.bilateralFilter(image, d=3, sigmaColor=75, sigmaSpace=75)

    mask = np.zeros(image.shape)     #  create a mask für original image
    mask = cv2.circle(mask, (mask.shape[0] // 2, mask.shape[1] // 2),
                      int(min(mask.shape[0], mask.shape[1]) * 0.5 * 0.95), (1, 1, 1), -1, 8, 0)
    mask = cv2.rectangle(mask, (0, 0), (mask.shape[0], 28), (0, 0, 0), -1, 8)
    mask = cv2.rectangle(mask, (0, 228), (mask.shape[0], mask.shape[1]), (0, 0, 0), -1, 8)
    image = image * mask

    #cv2.imshow("resized_image2", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return image


# make the image to quadrat
def letterbox(origin_image):
    height, width = origin_image.shape[:2]
    if height < width:
        # cv2.copyMakeBorder(origin_image, round(abs(width-height)/2), 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        return cv2.copyMakeBorder(origin_image, round(abs(width-height)/2), round(abs(width-height)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif height > width:
        # cv2.copyMakeBorder(origin_image, 0, 0, round(abs(width-height)/2), 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        return cv2.copyMakeBorder(origin_image, 0, 0, round(abs(width-height)/2), round(abs(width-height)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
    elif height == width:
        return origin_image


# need to fill with the dataset name
def read_path():
    if FLAGS.server:
        dataset_path='/home/data/IDRID_dataset/images/test'
        write_name_server = '/home/RUS_CIP/st181715/dl-lab-24w-team06/diabetic_retinopathy/idrid/IDRID_dataset/images/test'
        # test
        # write_name_server = '/home/kusabi/DLLAB/test_datasetpath'
        assert os.path.exists(write_name_server)  # check if the folder exist
        if os.listdir(write_name_server):  # check if the folder is empty
            print('write path is not empty, delete the old data')
            for filename in os.listdir(write_name_server):
                os.remove(write_name_server + '/' + filename)
        else:
            print('write path is empty, ready to propercess the data')

        for filename in os.listdir(dataset_path):
            origin_image = cv2.imread(dataset_path + '/' + filename)
            new_dataimage = Graham(origin_image)
            cv2.imwrite(write_name_server + '/' + filename, new_dataimage)
            print("finish " + filename)

        dataset_path_train='/home/data/IDRID_dataset/images/train'
        write_name_server_train = '/home/RUS_CIP/st181715/dl-lab-24w-team06/diabetic_retinopathy/idrid/IDRID_dataset/images/train'
        # test
        # write_name_server = '/home/kusabi/DLLAB/test_datasetpath'
        assert os.path.exists(write_name_server_train)  # check if the folder exist
        if os.listdir(write_name_server_train):  # check if the folder is empty
            print('write path is not empty, delete the old data')
            for filename in os.listdir(write_name_server_train):
                os.remove(write_name_server_train + '/' + filename)
        else:
            print('write path is empty, ready to propercess the data')

        for filename in os.listdir(dataset_path_train):
            origin_image = cv2.imread(dataset_path_train + '/' + filename)
            new_dataimage = Graham(origin_image)
            cv2.imwrite(write_name_server + '/' + filename, new_dataimage)
            print("finish " + filename)

    else:
        dataset_name = '/idrid/IDRID_dataset/images/test'
        write_name = '/new_dataset'
        filepath = os.getcwd() + dataset_name
        for filename in os.listdir(filepath):
            origin_image = cv2.imread(filepath + '/' + filename)
            #new_dataimage = preprocessing_filter_greenchannel(origin_image)  # this is the propercessing type
            #new_dataimage = histrogram_enhanced(origin_image)
            new_dataimage = histrogram_CLAHE(origin_image)
            #new_dataimage = Graham(origin_image)
            cv2.imwrite(os.getcwd() + write_name + '/' + filename, new_dataimage)
            print(os.getcwd() + write_name + '/' + filename)
            print("finish " + filename)

def main(_argv):
    print('preporcessing running.......')
    FLAGS.server = False
    read_path()
    print("All finish")


if __name__ == "__main__":
    try:
        app.run(main)
        print(cv2.__version__)
    except SystemExit:
        pass
