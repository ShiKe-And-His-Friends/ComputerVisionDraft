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
    with open(file_path ,"w" ,encoding = 'utf-8' ) as f:
        for token in vocab:
            # token = token.encode( encoding='utf-8')
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
    txt_tokens,
    separator = ' ' ,
    axis = -1
)
words = en_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(
    words,
    separator = ' ',
    axis = -1
)

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count ,1] ,START)
    ends = tf.fill([count ,1] ,END)
    return tf.concat([starts ,ragged ,ends] ,axis = 1)
words = en_tokenizer.detokenize(add_start_end(token_batch))
text_type =  tf.strings.reduce_join(words ,separator = ' ' ,axis = -1)
print(text_type)
def cleanup_text(reserved_tokens ,token_txt):
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_tokens_re = "|".join(bad_tokens)
    bad_cells = tf.strings.regex_full_match(token_txt ,bad_tokens_re)
    result = tf.ragged.boolean_mask(token_txt ,~bad_cells)
    result = tf.strings.reduce_join(result ,separator = ' ' ,axis = -1)
    return result
print(en_examples.numpy())
token_batch = en_tokenizer.tokenize(en_examples).merge_dims(-2 ,-1)
words = en_tokenizer.detokenize(token_batch)
print(words)
print(cleanup_text(reserved_tokens ,words).numpy())

class CustomTokenizer(tf.Module):
    def __init__(self ,reserved_tokens ,vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path ,lower_case = True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)
        file_data = pathlib.Path(vocab_path)
        file_text = file_data.read_text(encoding='utf-8')
        vocab = file_text.splitlines()
        self.vocab = tf.Variable(vocab)
        # create signature for export
        self.tokenize.get_concrete_function(
            tf.TensorSpec(
                shape = [None],
                dtype = tf.string
            )
        )
        self.detokenize.get_concrete_function(
            tf.TensorSpec(
                shape = [None ,None],
                dtype = tf.int64
            )
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(
                shape = [None ,None],
                dtype = tf.int64
            )
        )
        self.lookup.get_concrete_function(
            tf.TensorSpec(
                shape = [None ,None],
                dtype = tf.int64
            )
        )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(
                shape = [None ,None],
                dtype = tf.int64
            )
        )
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self ,strings):
        enc = self.tokenizer.tokenize(strings)
        enc = enc.merge_dims(-2 ,-1)
        enc = add_start_end(enc)
        return enc
    @tf.function
    def detokenize(self ,tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens ,words)
    @tf.function
    def lookup(self ,token_ids):
        return tf.gather(self.vocab ,token_ids)
    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]
    @tf.function
    def get_vocab_path(self):
        return self._vocab_path
    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)
tokenizers = tf.Module()
tokenizers.pt = CustomTokenizer(reserved_tokens ,"pt_vocab.txt")
tokenizers.en = CustomTokenizer(reserved_tokens ,"en_vocab.txt")
model_name = "ted_hrlr_translate_pt_en_converter"
tf.saved_model.save(tokenizers ,model_name)
reloaded_tokenizers = tf.saved_model.load(model_name)
reloaded_tokenizers.en.get_vocab_size().numpy()
tokens = reloaded_tokenizers.en.tokenize(['Hello Tensor Flows'])
print(tokens.numpy())
text_tokens = reloaded_tokenizers.en.lookup(tokens)
print(text_tokens)
round_trip = reloaded_tokenizers.en.detokenize(tokens)
print(round_trip.numpy()[0].decode('utf-8'))
pt_lookup = tf.lookup.StaticVocabularyTable(
    num_oov_buckets = 1,
    initializer = tf.lookup.TextFileInitializer(
        filename = 'pt_vocab.txt',
        key_dtype = tf.string,
        key_index = tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype = tf.int64,
        value_index = tf.lookup.TextFileIndex.LINE_NUMBER
    )
)
pt_tokenizer = text.BertTokenizer(pt_lookup)
pt_lookup.lookup(tf.constant(['e', 'um', 'uma', 'para', 'nao']))
pt_lookup = tf.lookup.StaticVocabularyTable(
    num_oov_buckets = 1,
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys = pt_vocab,
        values = tf.range(len(pt_vocab) ,dtype = tf.int64)
    )
)
pt_tokenizer = text.BertTokenizer(pt_lookup)

print("Input text tokenizer done.")
