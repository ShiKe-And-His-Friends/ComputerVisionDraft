import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images ,train_labels) ,(test_images ,test_labels) = fashion_mnist.load_data()
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
plt.figure(figsize = (10 ,10))
for i in range(25):
    plt.subplot(5 ,5 ,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
train_images.shape
test_images.shape

print("Image classify basic done.")
