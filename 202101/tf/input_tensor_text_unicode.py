# encoding:gbk
import tensorflow as tf
import numpy as np

#vim: set encoding = utf-8

print(tf.constant(u"Thanks  ÎÎÎÎ 高清"))
print(tf.constant([u"You're" ,u"welcome"]).shape)
text_utf8 =  tf.constant(u"a�˹�����")
print(text_utf8)
text_utf16be = tf.constant(u"���Դ���".encode("UTF-16-BE"))
print(text_utf16be)
text_char = tf.constant([ord(char) for char in u"���Դ���"])
print(text_char)
text_type = tf.strings.unicode_decode(text_utf8 ,input_encoding = 'UTF-8')
print(text_type)
text_type = tf.strings.unicode_encode(text_char ,output_encoding = 'UTF-8')
print(text_type)
text_type = tf.strings.unicode_transcode(text_utf8 ,input_encoding = 'UTF-8' ,output_encoding = 'UTF-16-BE')
print(text_type)
batch_utf8 = [
    s.encode('UTF-8') for s in 
    [u'h?llo', u'What is the weather tomorrow', u'G??dnight', u'��������']
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
print(
    '[%s]' % '\n'.join(
        '[%s]' % ','.join(
            value.rjust(max_width) for value in row
        )
        for row in elements
    )
)
text_type = tf.strings.unicode_encode(
    [[99, 97, 116], [100, 111, 103], [99, 111, 119]],
    output_encoding = 'UTF-8'
)
print(text_type)
text_type = tf.strings.unicode_encode(batch_chars_ragged ,output_encoding = 'UTF-8')
print(text_type)
text_type = tf.strings.unicode_encode(
    tf.RaggedTensor.from_sparse(batch_chars_sparse),
    output_encoding = 'UTF-8'
)
print(text_type)
text_type = tf.strings.unicode_encode(
    tf.RaggedTensor.from_tensor(batch_chars_padded ,padding = -1),
    output_encoding = 'UTF-8'
)
print(text_type)

thanks = u'Thanks ��������'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks ,unit='UTF8_CHAR').numpy()
print('{} bytes; {} UTF-8 characters'.format(num_bytes ,num_chars))
text_type = tf.strings.substr(thanks ,pos = 7 ,len = 1).numpy()
print(text_type)
print(tf.strings.substr(thanks ,pos = 7 ,len = 1 ,unit = 'UTF8_CHAR').numpy())
print(tf.strings.unicode_split(thanks ,'UTF-8').numpy())
codepoints ,offsets = tf.strings.unicode_decode_with_offsets(u'������������������������' ,'UTF-8')
for (codepoint ,offset) in zip(codepoints.numpy() ,offsets.numpy()):
    print('At byte offsets {}: codepoint {}.'.format(offset ,codepoint))
uscript = tf.strings.unicode_script([33464 ,1041])
print(uscript.numpy())
print(tf.strings.unicode_script(batch_chars_ragged))
sentence_texts = [u'Hello, world.', u'���礳��ˤ���']
sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts ,'UTF-8')
print(sentence_char_codepoint)
sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
print(sentence_char_script)
sentence_char_starts_word = tf.concat(
    [
        tf.fill([sentence_char_script.nrows() ,1] ,True),
        tf.not_equal(
            sentence_char_script[: , 1:],
            sentence_char_script[: , :-1]
        )
    ],
    axis = 1
)
word_starts = tf.squeeze(
    tf.where(sentence_char_starts_word.values),
    axis = 1
)
print(word_starts)
word_char_codepoint = tf.RaggedTensor.from_row_starts(
    values = sentence_char_codepoint.values,
    row_starts = word_starts
)
print(word_char_codepoint)
sentence_num_words = tf.reduce_sum(
    tf.cast(sentence_char_starts_word ,tf.int64),
    axis = 1
)
sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
    values = word_char_codepoint,
    row_lengths = sentence_num_words
)
print(sentence_word_char_codepoint)
tf.strings.unicode_encode(sentence_word_char_codepoint ,'UTF-8').to_list()

print("Input text unicodes type done.")
