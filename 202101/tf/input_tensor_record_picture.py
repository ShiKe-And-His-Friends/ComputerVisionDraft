import tensorflow as tf
import numpy as np
import IPython.display as display

cat_in_snow = tf.keras.utils.get_file(
    '320px-Felis_catus-cat_on_snow.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'
)
williamsburg_bridge = tf.keras.utils.get_file(
    '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'
)
display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))
image_labels = {
    cat_in_snow:0,
    williamsburg_bridge:1
}
image_string = open(cat_in_snow ,'rb').read()
label = image_labels[cat_in_snow]

def _bytes_feature(value):
    if isinstance(value ,type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def image_example(image_string ,label):
    image_shape = tf.io.decode_jpeg(image_string).shape
    feature = {
        'height':_int64_feature(image_shape[0]),
        'width':_int64_feature(image_shape[1]),
        'depth':_int64_feature(image_shape[2]),
        'label':_int64_feature(label),
        'image_raw':_bytes_feature(image_string),
    }
    return tf.train.Example(features = tf.train.Features(feature=feature))

for line in str(image_example(image_string ,label)).split('\n')[:15]:
    print(line)
print('...')

record_file = 'images.tfrecord'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename ,label in image_labels.items():
        images_string = open(filename ,"rb").read()
        tf_example = image_example(image_string ,label)
        writer.write(tf_example.SerializeToString())

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecord')
image_feature_description = {
    'height':tf.io.FixedLenFeature([] ,tf.int64),
    'width':tf.io.FixedLenFeature([] ,tf.int64),
    'depth':tf.io.FixedLenFeature([] ,tf.int64),
    'label':tf.io.FixedLenFeature([] ,tf.int64),
    'image_raw':tf.io.FixedLenFeature([] ,tf.string),
}
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto ,image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset)
for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    display.display(display.Image(data = image_raw))

print("Input tensor format piture data done.")
