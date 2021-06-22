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

print("\nTrain IMDB movice model done.")
