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
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
embedding_layer = tf.keras.layers.Embedding(1000 ,5)
result = embedding_layer(tf.constant([1 ,2 ,3]))
print(result.numpy())
result = embedding_layer(tf.constant([[0 ,1 ,2] ,[3 ,4 ,5]]))
print(result.shape)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase ,'<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )
vocab_size = 10000
sequence_length = 100
vectorize_layer = TextVectorization(
    standardize = custom_standardization,
    max_tokens = vocab_size,
    output_mode = 'int',
    output_sequence_length = sequence_length
)
text_ds = train_ds.map(lambda x ,y : x)
vectorize_layer.adapt(text_ds)
embedding_dim = 18
model = Sequential([
    vectorize_layer,
    Embedding(
        vocab_size,
        embedding_dim,
        name = "embedding"
    ),
    GlobalAveragePooling1D(),
    Dense(16 ,activation = 'relu'),
    Dense(1)
])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "logs")
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
    metrics = ['accuracy']
)
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 15,
    callbacks = [tensorboard_callback]
)
print(model.summary())

weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
out_v = io.open('vectors.tsv' ,'w' ,encoding = 'utf-8')
out_m = io.open('metadata.tsv' ,'w' ,encoding = 'utf-8')
for index ,word in enumerate(vocab):
    if index == 0:
        continue
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()
try:
    from google.colab import files
    files.download('vectors.tsv')
    files.download('metadata.tsv')
except Exception:
    pass

print("Text Embedding done.")
