import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

#download data
print(tf.__version__)
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(
    origin = dataset_url,
    fname = 'flower_photos',
    untar = True
)
data_dir = pathlib.Path(data_dir)
print(data_dir)
image_count = len(list(data_dir.glob('*\*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses\*'))
PIL.Image.open(str(roses[0]))

# preprocess data
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
class_name = train_ds.class_names
print(class_name)
plt.figure(figsize = (10 ,10))
for images ,labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3 ,3 , i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_name[labels[i]])
        plt.axis("off")
# plt.show()
for image_batch ,labels_batch in train_ds:
    print(image_batch)
    print(labels_batch)
    break
# standardize the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalization_ds = train_ds.map(lambda x ,y : (normalization_layer(x) ,y))
image_batch ,labels_batch = next(iter(normalization_ds))
first_image = image_batch[0]
print(np.min(first_image) ,np.max(first_image))

# train model
AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
train_ds = train_ds.take(AUTOTUNE).cache().repeat()
val_ds = val_ds.take(AUTOTUNE).cache().repeat()

num_classes = 5
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(32 ,3 ,activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32 ,3 ,activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32 ,3 ,activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128 ,activation = 'relu'),
    tf.keras.layers.Dense(num_classes)
])
model.compile(
    optimizer = 'adam',
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']

)
'''
jpeg::Uncompress failed.
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)
'''

# finer control
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*') ,shuffle = False)
list_ds = list_ds.shuffle(image_count ,reshuffle_each_iteration = False)
for f in list_ds.take(5):
    print(f.numpy())
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
    parts = tf.strings.split(file_path ,os.path.sep)
    print(parts)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def decode_img(img):
    img= tf.io.decode_jpeg(img ,channels = 3)
    return tf.image.resize(img ,[img_height ,img_width])

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img ,label

train_ds = train_ds.map(process_path ,num_parallel_calls = AUTOTUNE)
val_ds = val_ds.map(process_path ,num_parallel_calls = AUTOTUNE)
for image ,label in train_ds.take(1):
    print("Image shape : " ,image.numpy().shape)
    print("Label: ",label.numpy())

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size = 1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

image_batch ,label_batch = next(iter(train_ds))
plt.figure(figsize = (10 ,10))
for i in range(9):
    ax = plt.subplot(3 ,3 , i+1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.title(class_names[label])
    plt.axis("off")
# plt.show()

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)

(train_ds ,val_ds ,test_ds) ,metadata = tfds.load(
    'tf_flowers',
    split = ['train[:80%]' ,'train[80%:90%]' ,'train[90%:]'],
    with_info = True,
    as_supervised = True,
)
num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str
image ,label = next(iter(train_ds))
plt.imshow(image)
plt.title(get_label_name(label))
plt.show()

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

print("input picture done.")
