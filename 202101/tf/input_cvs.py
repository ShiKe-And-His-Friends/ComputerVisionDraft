import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import itertools
import pathlib
import re
from matplotlib import pyplot as plt

np.set_printoptions(precision = 3 ,suppress = True)

# download data
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names = ["Length" ,"Diameter" ,"Height" ,"Whole weight" ,"Shucked weight"
        ,"Viscera weight" ,"Shell weight" ,"Age"]
)
print(abalone_train.head())
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')
abalone_features = np.array(abalone_features)
print(abalone_labels)
print(abalone_features)

# simple train
abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])
abalone_model.compile(
    loss = tf.losses.MeanSquaredError(),
    optimizer = tf.optimizers.Adam()
)
abalone_model.fit(
    abalone_features,
    abalone_labels,
    epochs = 10
)
normalize = preprocessing.Normalization()
print("\n\n\n 0:\n")
print(abalone_features)
normalize.adapt(abalone_features)
norm_abalone_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
])
norm_abalone_model.compile(
    loss = tf.losses.MeanSquaredError(),
    optimizer = tf.optimizers.Adam()
)
norm_abalone_model.fit(
    abalone_features,
    abalone_labels,
    epochs = 10
)

# predict tatanic survivy
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
print(titanic.head())
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop("survived")
input = tf.keras.Input(shape =() ,dtype = tf.float32)
result = 2 * input + 1
print(result)
calc = tf.keras.Model(inputs = input ,outputs = result)
print(calc(1).numpy())
print(calc(2).numpy())

inputs = {}
for name ,column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape = (1 ,) ,name = name ,dtype = dtype)
print(inputs)
numeric_inputs = {name:input for name ,input in inputs.items()
    if input.dtype == tf.float32}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
print(all_numeric_inputs)
preprocessed_inputs = [all_numeric_inputs]

for name ,input in inputs.items():
    if input.dtype == tf.float32:
        continue
    lookup = preprocessing.StringLookup(vocabulary = np.unique(titanic_features[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens = lookup.vocab_size())

    x= lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)
print(preprocessed_inputs)
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs ,preprocessed_inputs_cat)

# draw model layer png
tf.keras.utils.plot_model(model = titanic_preprocessing ,rankdir = "LR" ,dpi = 72 ,show_shapes = True)

titanic_features_dict = {name : np.array(value) 
    for name ,value in titanic_features.items()}
titanic_preprocessing(titanic_features_dict)
feature_dict = {name:values[:1] for name ,values in titanic_features_dict.items()}
print(titanic_preprocessing(feature_dict))

def titanic_model(preprocessing_head ,inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs ,result)
    model.compile(loss = tf.losses.BinaryCrossentropy(from_logits = True),
        optimizer = tf.optimizers.Adam())
    return model
titanic_model = titanic_model(titanic_preprocessing ,inputs)
titanic_model.fit(
    x = titanic_features_dict,
    y = titanic_labels,
    epochs = 10
)
titanic_model.save('titanic_test')
reloaded = tf.keras.models.load_model('titanic_test')
feature_dict = {name:values[:1] for name ,values in titanic_features_dict.items()}
before = titanic_model(feature_dict)
after = reloaded(feature_dict)
assert(before - after) < 1e-3
print(before)
print(after)

# in memory data
def slices(features):
    for i in itertools.count():
        example = {name:values[i] for name ,values in features.items()}
        yield example
for example in slices(titanic_features_dict):
    for name ,value in example.items():
        print(f"{name:19s}:{value}")
    break
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict ,titanic_labels))
titanic_batchs = titanic_ds.shuffle(len(titanic_labels)).batch(32)
titanic_model.fit(
    titanic_batchs,
    epochs = 5
)

titanic_file_path = tf.keras.utils.get_file("train.csv" ,"https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_csv_ds = tf.data.experimental.make_csv_dataset(
    titanic_file_path,
    batch_size = 5,
    label_name = 'survived',
    num_epochs = 1,
    ignore_errors = True,
)
for batch ,label in titanic_csv_ds.take(1):
    for key ,value in batch.items():
        print(f"{key:20s}:{value}")
    print()
    print(f"{'label':20s}:{label}")

traffic_volume_csv_gz = tf.keras.utils.get_file(
    'Metro_Interstate_Traffic_Volume.csv.gz',
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    cache_dir = '.',
    cache_subdir = 'traffic'
)
traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
    traffic_volume_csv_gz,
    batch_size = 256,
    label_name = 'traffic_volume',
    num_epochs = 1,
    compression_type = "GZIP"
)
for batch ,label in traffic_volume_csv_gz_ds.take(1):
    for key ,value in batch.items():
        print(f"{key:20s}:{value[:5]}")
    print()
    print(f"{'label':20s}:{label[:5]}")
for i ,(batch ,label) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
    if i % 40 == 0:
        print('.',end = '')
print()
snapshot = tf.data.experimental.snapshot('titanic.tfsnap')
snapshotting = traffic_volume_csv_gz_ds.apply(snapshot).shuffle(1000)
for i ,(batch ,label) in enumerate(snapshotting.shuffle(1000).repeat(20)):
    if i % 40 == 0:
        print('.' ,end='')
print()

# multiple character
font_zip = tf.keras.utils.get_file(
    'font.zip',
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir = '.',
    cache_subdir = 'fonts',
    extract = True
)
fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size = 10,
    num_epochs = 1,
    num_parallel_reads = 20,
    shuffle_buffer_size = 10000
)
font_csvs = sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))
print(font_csvs[:10])
print(len(font_csvs))
for features in fonts_ds.take(1):
    for i ,(name ,value) in enumerate(features.items()):
        if i > 15:
            break
        print(f"{name:20s}:{value}")
print('...')
print(f"[total: {len(features)} features]")

def make_images(features):
    image = [None] * 400
    new_feats = {}
    for name ,value in features.items():
        match = re.match('r(\d+)c(\d+)' ,name)
        if match:
            image[int(match.group(1)) * 20 + int(match.group(2))] = value
        else:
            new_feats[name] = value
    image = tf.stack(image ,axis = 0)
    image = tf.reshape(image ,[20 ,20 ,-1])
    new_feats['image'] = image
    return new_feats
fonts_image_ds = fonts_ds.map(make_images)
for features in fonts_image_ds.take(1):
    break
plt.figure(figsize = (6,6) ,dpi = 120)
for n in range(9):
    plt.subplot(3 ,3 , n+1)
    plt.imshow(features['image'][... ,n])
    plt.title(chr(features['m_label'][n]))
    plt.axis('off')
plt.show()

# basic level functions


print("\nInput CVS data done.\n")
