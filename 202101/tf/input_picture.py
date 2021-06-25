import tensorflow as tf
import pathlib
import random
from IPython.display import display ,Image
import matplotlib.pyplot as plt

# download photo
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_root_orig = tf.keras.utils.get_file(
    origin = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    fname = 'flower_photos',
    untar = True
)
data_root = pathlib.Path(data_root_orig)
print(data_root)
for item in data_root.iterdir():
    print(item)

# local photo add tag
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(all_image_paths[:10])
print(image_count)

attributions = (data_root/"LICENSE.txt").open(encoding = 'utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)
# print(attributions)

def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel).replace("\\","/")].split(' - ')[:-1])

for n in range(3):
    image_path = random.choice(all_image_paths)
    path_windows = str(image_path)
    path_windows = path_windows.replace("\\" ,"/")
    # display(Image(filename = image_path)) #only juypter notebook
    plt.imshow(plt.imread(path_windows))
    # plt.show() # debug show
    print(caption_image(path_windows))
    print()

# tag reference
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
label_to_index = dict((name ,index) for index ,name in enumerate(label_names))
print(label_to_index)
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
        for path in all_image_paths]
print("First 10 labels indices :" ,all_image_labels[:10])

# tranfer format
img_path = all_image_paths[0]
print(img_path)
img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100] + "...")
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)
img_final = tf.image.resize(img_tensor ,[192 ,192])
img_final = img_final / 255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

def preprocess_image(image):
    image = tf.image.decode_jpeg(image ,channels = 3)
    image = tf.image.resize(image ,[192 ,192])
    image /= 255.0 # normalize
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

image_path = all_image_paths[0]
label = all_image_labels[0]
plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
# plt.show()
print()

# contibute a dataset
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)
images_ds = path_ds.map(load_and_preprocess_image ,num_parallel_calls = AUTOTUNE)
plt.figure(figsize = (8 ,8))
for n ,image in enumerate(images_ds.take(4)):
    plt.subplot(2 ,2 ,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))
    # plt.show()
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels ,tf.int64))
for label in label_ds.take(10):
    print(label_names[label.numpy()])
image_label_ds = tf.data.Dataset.zip((images_ds ,label_ds))
print(image_label_ds)
ds = tf.data.Dataset.from_tensor_slices((all_image_paths ,all_image_labels))

def load_and_preprocess_from_path_label(path ,label):
    return load_and_preprocess_image(path) ,label
image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

# manifesit dataset
BATCH_SIZE = 32
ds = image_label_ds.shuffle(buffer_size = image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size = AUTOTUNE)
print(ds)
ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size = image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size = AUTOTUNE)
print(ds)

# hub model
mobile_net = tf.keras.applications.MobileNetV2(input_shape = (192 ,192 ,3) ,include_top = False)
mobile_net.trainable = False
help(mobile_net.preprocess_input)

print("input picture done.")
