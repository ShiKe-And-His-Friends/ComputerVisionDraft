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
binary_loss ,binary_accuracy = binary_model.evaluate(binary_test_ds)
int_loss ,int_accuracy = int_model.evaluate(int_test_ds)
print("Binary model accuray:{:2.2%}".format(binary_accuracy))
print("Int model accuracy:{:2.2%}".format(int_accuracy))

export_model = tf.keras.Sequential([
    binary_vectorize_layer,
    binary_model,
    layers.Activation('sigmoid')
])
export_model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits = False),
    optimizer = 'adam',
    metrics = ['accuracy']
)
loss ,accuracy = export_model.evaluate(raw_test_ds)
print("Accuracy:{:2.2%}".format(binary_accuracy))
def get_string_labels(predicted_scores_batch):
    predicted_int_labels = tf.argmax(predicted_scores_batch ,axis =1)
    predicted_labels = tf.gather(raw_train_ds.class_names ,predicted_int_labels)
    return predicted_labels
inputs = [
    "how do I extract keys from a dict into a list?",
    "debug public static void main(string[] ){...}"
]
predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)
for input ,label in zip(inputs ,predicted_labels):
    print("Question:" ,input)
    print("Predicted labels:" ,label.numpy())
DIRECTORY_URL = "https://storage.googleapis.com/download.tensorflow.org/data/illiad/"
FILE_NAMES = ['cowper.txt' ,'derby.txt' ,'butler.txt']
for name in FILE_NAMES:
    text_dir = utils.get_file(name ,origin = DIRECTORY_URL + name)
parent_dir = pathlib.Path(text_dir).parent
print(list(parent_dir.iterdir()))
def labeler(example ,index):
    return example ,tf.cast(index ,tf.int64)
labeled_data_sets = []
for i,file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(str(parent_dir/file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex ,i))
    labeled_data_sets.append(labeled_dataset)
BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE,
    reshuffle_each_iteration = False
)
for text ,label in all_labeled_data.take(10):
    print("Sentence:" ,text.numpy())
    print("Label:" ,label.numpy())
tokenizer = tf_text.UnicodeScriptTokenizer()
def tokenize(text ,unused_label):
    lower_case = tf_text.case_fold_utf8(text)
    return tokenizer.tokenize(lower_case)
tokenized_ds = all_labeled_data.map(tokenize)
for text_batch in tokenized_ds.take(5):
    print("Token:" ,text_batch.numpy())
tokenized_ds = configure_dataset(tokenized_ds)
vocab_dict = collections.defaultdict(lambda: 0)
for toks in tokenized_ds.as_numpy_iterator():
    for tok in toks:
        vocab_dict[tok] += 1
vocab = sorted(vocab_dict.items() ,key = lambda x : x[1] ,reverse = True)
vocab = [token for token ,count in vocab]
vocab_size = len(vocab)
print("Vocab size:" ,vocab_size)
print("First five vocab entries:" ,vocab[:5])
keys = vocab
values = range(2 ,len(vocab) + 2)
init = tf.lookup.KeyValueTensorInitializer(
    keys,
    values,
    key_dtype = tf.string,
    value_dtype = tf.int64
)
num_oov_buckets = 1
vocab_table = tf.lookup.StaticVocabularyTable(init ,num_oov_buckets)
def preprocess_text(text ,label):
    standardized = tf_text.case_fold_utf8(text)
    tokenized = tokenizer.tokenize(standardized)
    voctorized = vocab_table.lookup(tokenized)
    return voctorized ,label
example_text ,example_label = next(iter(all_labeled_data))
print("Sentence:" ,example_text.numpy())
vectorized_text ,example_label = preprocess_text(example_text ,example_label)
print("Vectorized sentence:" ,vectorized_text.numpy())
all_encoded_data = all_labeled_data.map(preprocess_text)

train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
validation_data = all_encoded_data.take(VALIDATION_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)
validation_data = validation_data.padded_batch(BATCH_SIZE)
sample_text ,sample_labels = next(iter(validation_data))
print("Text batch shape:" ,sample_text.shape)
print("Label batch shape:" ,sample_labels.shape)
print("First text example:" ,sample_text[0])
print("First Label example:" ,sample_labels[0])
vocab_size += 2
train_data = configure_dataset(train_data)
validation_data = configure_dataset(validation_data)
model = create_model(
    vocab_size = vocab_size ,
    num_labels = 3
)
model.compile(
    optimizer = 'adam',
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)
history = model.fit(
    train_data,
    validation_data = validation_data,
    epochs = 3
)
loss ,accuracy = model.evaluate(validation_data)
print("Loss:" ,loss)
print("Accuracy:{:2.2%}".format(accuracy))
preprocess_layer = TextVectorization(
    max_tokens = vocab_size,
    standardize = tf_text.case_fold_utf8,
    split = tokenizer.tokenize,
    output_mode = 'int',
    output_sequence_length = MAX_SEQUENCE_LENGTH
)
preprocess_layer.set_vocabulary(vocab)
export_model = tf.keras.Sequential([
    preprocess_layer,
    model,
    layers.Activation('sigmoid')
])
export_model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits = False),
    optimizer = 'adam',
    metrics = ['accuracy']
)
test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
test_ds = configure_dataset(test_ds)
loss ,accuracy = export_model.evaluate(test_ds)
print("Losses:" ,loss)
print("Accuracy:{:2.2%}".format(accuracy))
inputs = [
    "Join'd to th' Ionians with their flowing robes,",
    "the allies, and his armour flashed about him so that he seemed to all",
    "And with loud clangor of his arms he fell."
]
predicted_scores = export_model.predict(inputs)
predicted_labels = tf.argmax(predicted_scores , axis = 1)
for input ,label in zip(inputs ,predicted_labels):
    print("Question:" ,input)
    print("Predicted label:" ,label.numpy())
train_ds = tfds.load(
    'imdb_reviews',
    split = 'train[:80%]',
    batch_size = BATCH_SIZE,
    shuffle_files = True,
    as_supervised = True
)
val_ds = tfds.load(
    'imdb_reviews',
    split = 'train[80%:]',
    batch_size = BATCH_SIZE,
    shuffle_files = True,
    as_supervised = True
)
for review_batch ,label_batch in val_ds.take(1):
    for i in range(5):
        print("Review:" ,review_batch[i].numpy())
        print("Label:" ,label_batch[i].numpy())
vectorize_layer = TextVectorization(
    max_tokens = VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length = MAX_SEQUENCE_LENGTH
)
train_text = train_ds.map(lambda text ,labels : text)
vectorize_layer.adapt(train_text)
def vectorize_text(text ,label):
    text = tf.expand_dims(text ,-1)
    return vectorize_layer(text) ,label
train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)
model = create_model(
    vocab_size = VOCAB_SIZE + 1,
    num_labels = 1
)
print(model.summary())
model.compile(
    loss = losses.BinaryCrossentropy(from_logits = True),
    optimizer = 'adam',
    metrics = ['accuracy']
)
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)
print(history)
loss ,accuracy = model.evaluate(val_ds)
print("Losses:" ,loss)
print("Accuracy:{:2.2%}".format(accuracy))
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])
export_model.compile(
    loss = losses.SparseCategoricalCrossentropy(from_logits = False),
    optimizer = 'adam',
    metrics = ['accuracy']
)
inputs = [
    "This is a fantastic movie.",
    "This is a bad movie.",
    "This movie was so bad that it was good.",
    "I will never say yes to watching this movie."
]
predicted_scores = export_model.predict(inputs)
predict_labels = [int(round(x[0])) for x in predicted_scores]
for input ,label in zip(inputs ,predicted_labels):
    print("Question:",input)
    print("Predicted label:" ,label)

print("Input text codes")
