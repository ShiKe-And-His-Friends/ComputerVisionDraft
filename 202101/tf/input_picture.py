import tensorflow as tf
import pathlib
import random

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

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(image_count)

print("input picture done.")
