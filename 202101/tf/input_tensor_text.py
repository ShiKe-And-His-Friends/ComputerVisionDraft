import collections
import pathlib
import re
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_datasets as tfds
import tensorflow_text as tf_text

data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset_dir = utils.get_file(
    fname = 'stack_overflow_16k',
    origin = data_url,
    untar = True,
    cache_dir = 'stack_overflow_text',
    cache_subdir = ''
)
dataset_dir = pathlib.Path(dataset_dir).parent
print(list(dataset_dir.iterdir()))
train_dir = dataset_dir/'train'
print(list(train_dir.iterdir()))
sample_file = train_dir/'python/1755.txt'
with open(sample_file) as f:
    print(f.read())
batch_size = 32
seed = 42
raw_train_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'training',
    seed = seed
)
for text_batch ,label_batch in raw_train_ds.take(1):
    for i in range(10):
        print("Question:" ,text_batch.numpy()[i])
        print("Label:" ,label_batch.numpy()[i])
for i ,label in enumerate(raw_train_ds.class_names):
    print("Label" ,i  ,"corresponds to" ,label)
raw_val_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'validation',
    seed = seed
)
test_dir = dataset_dir/'test'
raw_test_ds = preprocessing.text_dataset_from_directory(
    test_dir,
    batch_size = batch_size
)

VOCAB_SIZE = 10000
binary_vectorize_layer = TextVectorization(
    max_tokens = VOCAB_SIZE,
    output_mode = 'binary'
)
MAX_SEQUENCE_LENGTH = 250
int_vectorize_layer = TextVectorization(
    max_tokens = VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length = MAX_SEQUENCE_LENGTH
)
train_text = raw_train_ds.map(lambda text ,labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)

def binary_vectorize_text(text ,label):
    text = tf.expand_dims(text ,-1)
    return binary_vectorize_layer(text) ,label
def int_vectorize_text(text ,label):
    text = tf.expand_dims(text ,-1)
    return int_vectorize_layer(text) ,label
text_batch ,label_batch = next(iter(raw_train_ds))
first_question ,first_label = text_batch[0] ,label_batch[0]
print("Question" ,first_question)
print("Label" ,first_label)
print("'binary' vectorized question:" ,binary_vectorize_text(first_question ,first_label)[0])
print("'int' vectorized question:" ,int_vectorize_text(first_question ,first_label)[0])
print("128 --> ",int_vectorize_layer.get_vocabulary()[1289])
print("313 --> ",int_vectorize_layer.get_vocabulary()[313])
print("Vocabulary size:{}".format(len(int_vectorize_layer.get_vocabulary())))
binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)
binary_test_ds = raw_test_ds.map(binary_vectorize_text)
int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size = AUTOTUNE)
binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)
binary_train_ds = configure_dataset(binary_test_ds)
int_tiran_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)
binary_model = tf.keras.Sequential([layers.Dense(4)])
binary_model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = 'adam',
    metrics = ['accuracy']
)
history = binary_model.fit(
    binary_train_ds,
    validation_data = binary_val_ds,
    epochs = 10
)
def create_model(vocab_size ,num_labels):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size ,64 ,mask_zero = True),
        layers.Conv1D(64 ,5 ,padding = "valid" ,activation = "relu" ,strides = 2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(num_labels)
    ])
    return model
int_model = create_model(vocab_size = VOCAB_SIZE + 1 ,num_labels = 4)
int_model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = 'adam',
    metrics = ['accuracy']
)
history = int_model.fit(
    int_train_ds,
    validation_data = int_val_ds,
    epochs = 5
)
print("Linear model on binary vectorized data:")
print(binary_model.summary())
print("ConvNet model on int vectorized data:")
print(int_model.summary())

print("Input text codes")
