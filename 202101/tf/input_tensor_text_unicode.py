# encoding:gbk
import tensorflow as tf
import numpy as np

#vim: set encoding = utf-8

print(tf.constant(u"Thanks  脦脦脦脦 楂樻竻"))
print(tf.constant([u"You're" ,u"welcome"]).shape)
text_utf8 =  tf.constant(u"a人工智能")
print(text_utf8)
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
print(text_utf16be)
text_char = tf.constant([ord(char) for char in u"语言处理"])
print(text_char)
text_type = tf.strings.unicode_decode(text_utf8 ,input_encoding = 'UTF-8')
print(text_type)
text_type = tf.strings.unicode_encode(text_char ,output_encoding = 'UTF-8')
print(text_type)
text_type = tf.strings.unicode_transcode(text_utf8 ,input_encoding = 'UTF-8' ,output_encoding = 'UTF-16-BE')
print(text_type)
batch_utf8 = [
    s.encode('UTF-8') for s in 
    [u'h?llo', u'What is the weather tomorrow', u'G??dnight', u'熙伍熙伍']
]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8 ,input_encoding = 'UTF-8')
for sentence_chars in batch_chars_ragged.to_list():
    print(sentence_chars)
batch_chars_padded = batch_chars_ragged.to_tensor(default_value = -1)
print(batch_chars_padded)
batch_chars_padded = batch_chars_ragged.to_tensor(default_value = -1)
print(batch_chars_padded)
batch_chars_sparse = batch_chars_ragged.to_sparse()
nrows ,ncols = batch_chars_sparse.dense_shape.numpy()
elements = [['_' for i in range(ncols)] for j in range(nrows)]
for (row ,col) ,value in zip(batch_chars_sparse.indices.numpy() ,batch_chars_sparse.values.numpy()):
    elements[row][col] = str(value)
value_lengths = []
for row in elements:
    for value in row:
        value_lengths.append(len(value))
max_width = max(value_lengths)
print('[%s]' % '\n'.join(
    '[%s]' % ','.join(value.rjust(max_width) for value in row)
    for row in elements))
text_type = tf.strings.unicode_encode(
    [[99, 97, 116], [100, 111, 103], [99, 111, 119]],
    output_encoding = 'UTF-8'
)
print(text_type)
text_type = tf.strings.unicode_encode(batch_chars_ragged ,output_encoding = 'UTF-8')
print(text_type)


print("Input text unicodes type done.")
