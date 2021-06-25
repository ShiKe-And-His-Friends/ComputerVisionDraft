import tensorflow as tf
import pathlib
import random
from IPython.display import display ,Image
import matplotlib.pyplot as plt

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
    # only juypter notebook
    # display(Image(filename = image_path))
    plt.imshow(plt.imread(path_windows))
    plt.show()
    print(caption_image(path_windows))
    print()

print("input picture done.")
