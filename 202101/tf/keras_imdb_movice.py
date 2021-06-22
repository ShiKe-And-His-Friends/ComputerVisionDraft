import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb
(train_data ,train_labels) ,(test_data ,test_labels) = imdb.load_data(num_words = 10000)

print("Training entries:{} ,labels:{}".format(len(train_data) ,len(train_labels)))
print(train_data[0])
print("One baseline legth:{},Two baseline length:{}".format(len(train_data[0]) ,len(train_data[1])))

# get words' dictionary
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["UNK"] = 2 # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value ,key) for (key ,value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i ,'?') for i in text])
text1 = decode_review(train_data[0])
print(text1)

train_data = keras.preprocessing.sequence.pad_sequences(train_data ,
        value = word_index["<PAD>"],
        padding = 'post',
        maxlen = 256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data ,
        value = word_index["<PAD>"],
        padding = 'post',
        maxlen = 256)
print("After prepare one baseline length:{},two baseline length:{}".format(len(train_data[0] ) ,len(train_data[1])))
prepare_text1 = decode_review(train_data[0])
print(prepare_text1)

# contribute model
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size ,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16 ,activation = 'relu'))
model.add(keras.layers.Dense(1 ,activation = 'sigmoid'))
model.summary()

model.compile(optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train model
history = model.fit(partial_x_train ,
        partial_y_train,
        epochs = 40,
        batch_size = 512,
        validation_data = (x_val ,y_val),
        verbose = 1)

print("\nTrain IMDB movice model done.")
