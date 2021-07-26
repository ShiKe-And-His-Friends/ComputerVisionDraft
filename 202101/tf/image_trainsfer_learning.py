import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file('cast_and_dogs.zip' ,origin = _URL ,extract = True)
PATH = os.path.join(os.path.dirname(path_to_zip) ,'cats_and_dogs_filtered')
train_dir = os.path.join(PATH ,'train')
validation_dir = os.path.join(PATH ,'validation')
BATCH_SIZE = 2
IMG_SIZE = (160 ,160)
train_dataset = image_dataset_from_directory(
    train_dir,
    shuffle = True,
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE
)
validation_dataset = image_dataset_from_directory(
    validation_dir,
    shuffle = True,
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE
)

print("Image transfer learning done,")
