import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images ,train_labels) ,(test_images ,test_labels) = fashion_mnist.load_data()

# print(len(train_labels))

class_names = ['T-shirt/top' ,'Trouser' ,'Pullover' ,'Dress' ,'Coat' ,'Sandal' ,'Shirt'
        ,'Sneaker' ,'Bag' ,'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

'''
plt.figure(figsize = (10 ,10))
for i in range(25):
    plt.subplot(5 ,5 ,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i] ,cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28 ,28)),
    keras.layers.Dense(128 ,activation = 'relu'),
    keras.layers.Dense(10)
    ])
model.compile(optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy'])
print("train fashion mnist model:")
model.fit(train_images ,train_labels ,epochs = 10)
print("Test fashion mnist model:")
test_loss ,test_acc = model.evaluate(test_images ,test_labels ,verbose = 2)
print("\naccuracy:" ,test_acc)

print("compile fashion mnist model done")

