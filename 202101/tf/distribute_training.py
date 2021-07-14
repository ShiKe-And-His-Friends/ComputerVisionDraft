import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os

print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images ,train_labels) ,(test_images ,test_labels) = fashion_mnist.load_data()
train_images = train_images[...,None]
test_images = test_images[...,None]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
strategy = tf.distribute.MirroredStrategy()

BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10
train_dataset = tf.data.Dataset.from_tensor_slices((train_images ,train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images ,test_labels)).batch(GLOBAL_BATCH_SIZE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

print("Distribute train done.")
