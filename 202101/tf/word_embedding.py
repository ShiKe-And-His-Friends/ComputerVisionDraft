import io
import os 
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense ,Embedding ,GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file(
    "aclImdb_v1.tar.gz",
    url,
    untar = True,
    cache_dir = '.',
    cache_subdir = ''
)
dataset_dir = os.path.join(os.path.dirname(dataset) ,'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir ,'train')
print(os.listdir(train_dir))
remove_dir = os.path.join(train_dir ,'unsup')
shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'training',
    seed = seed
)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'validation',
    seed = seed
)
for text_batch ,label_batch in train_ds.take(1):
    for i in range(5):
        print(label_batch[i].numpy() ,text_batch.numpy()[i])

print("Text Embedding done.")
