import tensorflow as tf
from tensorflow.keras import datasets ,layers ,models
import matplotlib.pyplot as plt

(train_images ,train_labels),(test_images ,test_labels) = datasets.cifar10.load_data()

model = models.Sequential()
model.add(layers.Conv2D(32 ,(3 ,3) ,avtivation = 'relu' ,input_shape = (32 ,32 ,3)))
