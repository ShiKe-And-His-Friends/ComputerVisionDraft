import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

examples ,metadata = tfds.load(
    'ted_hrlr_translate/pt_to_en',
    with_info = True,
    as_supervised = True
)
train_examples ,val_examples = examples['train'] ,examples['validation']
for pt,en in train_examples.take(1):
    print("Portuguese:" ,pt.numpy().decode('utf-8'))
    print("English:" ,en.numpy().decode('utf-8'))
train_en = train_examples.map(lambda pt, en: en)
train_pt = train_examples.map(lambda pt, en: pt)
bert_tokenizer_params = dict(lower_case = True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
bert_vocab_args = dict(
    vocab_size = 8000,
    reserved_tokens = reserved_tokens,
    bert_tokenizer_params = bert_tokenizer_params,
    learn_params = {},
)
pt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_pt.batch(1000).prefetch(2),
    **bert_vocab_args
)
print(pt_vocab[:10])
print(pt_vocab[100:110])
print(pt_vocab[1000:1010])
print(pt_vocab[-10:])
def write_vocab_file(file_path ,vocab):
    with open(file_path ,"w" ,encoding = 'UTF-8') as f:
        for token in vocab:
            print(token ,file = f)
write_vocab_file(u'pt_vocab.txt' ,pt_vocab)
en_vocab = bert_vocab.bert_vocab_from_dataset(
    train_en.batch(1000).prefetch(2),
    **bert_vocab_args
)
print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])
write_vocab_file('en_vocab.txt' ,en_vocab)

pt_tokenizer = text.BertTokenizer('pt_vocab.txt' ,**bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt' ,**bert_tokenizer_params)
for pt_examples ,en_examples in train_examples.batch(3).take(1):
    for ex in en_examples:
        print(ex.numpy())
token_batch = en_tokenizer.tokenize(en_examples)
token_batch = token_batch.merge_dims(-2 ,-1)
for ex in token_batch.to_list():
    print(ex)
txt_tokens = tf.gather(en_vocab ,token_batch)
tf.strings.reduce_join(
    txt_tokens ,
    separator = ' ' ,
    axis = -1
)
words = en_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(
    words,
    separator = ' ',
    axis = -1
)

print("Input text tokenizer done.")
