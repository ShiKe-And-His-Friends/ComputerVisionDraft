import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import tensorflow as tf
from tensorflow.keras import layers ,Sequential
import pathlib

print(tf.__version__)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos' ,origin = dataset_url ,untar = True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
print(str(roses[0]))

tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height ,img_width),
    batch_size = batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height ,img_width),
    batch_size = batch_size
)
class_names = train_ds.class_names
print(class_names)

plt.figure(figsize = (10 ,10))
for images ,labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3 ,3 ,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()
for image_batch ,labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
normallization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x,y : (normallization_layer(x),y))
image_batch ,labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image) ,np.max(first_image))

num_classes = 5
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255 ,input_shape = (img_height ,img_width ,3)),
    layers.Conv2D(16 ,3 ,padding = 'same' ,activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32 ,3 ,padding = 'same' ,activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64 ,3 ,padding = 'same' ,activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128 ,activation = 'relu'),
    layers.Dense(num_classes)
])
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)
print(model.summary())
epochs = 10
history = model.fit(
    train_ds,
    validation_data =  val_ds,
    epochs = epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize = (8 ,8))
plt.subplot(1 ,2  ,1)
plt.plot(epochs_range ,acc ,label = 'Training Accuracy')
plt.plot(epochs_range ,val_acc ,label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1 ,2  ,2)
plt.plot(epochs_range ,loss ,label = 'Training Loss')
plt.plot(epochs_range ,val_loss ,label = 'Validations Loss')
plt.legend(loc = 'upper right')
plt.title('Training and validation Loss')
# plt.show()

data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal" ,input_shape = (img_height ,img_width ,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])
plt.figure(figsize = (10 ,10))
for images ,_ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3 ,3 ,i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
# plt.show()

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16 ,3 ,padding = 'same' ,activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32 ,3 ,padding = 'same' ,activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64 ,3 ,padding = 'same' ,activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128 ,activation = 'relu'),
    layers.Dense(num_classes)
])
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)
print(model.summary())
epochs = 15
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)

print("Image classify basic done.")
