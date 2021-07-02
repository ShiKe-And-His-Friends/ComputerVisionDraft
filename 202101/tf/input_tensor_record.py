import tensorflow as tf
import numpy as np
import IPython.display as display

def _bytes_feature(value):
    """Return  a bytes_list from a string / byte."""
    if isinstance(value ,type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    """Return a float_list from a float / double ."""
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def _int64_feature(value):
    """Return a int64_list from a bool / enum / uint ."""
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

print(_bytes_feature(b'test_strting'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))
print(_float_feature(np.exp(1)))
print(_int64_feature(True))
print(_int64_feature(1))
feature = _float_feature(np.exp(1))
print(feature.SerializeToString())

n_observations = int(1e4)
feature0 = np.random.choice([False ,True] ,n_observations)
feature1 = np.random.randint(0 ,5 ,n_observations)
strings = np.array([b'cat' ,b'dog' ,b'chicken' ,b'horse' ,b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

def serialize_example(feature0 ,feature1 ,feature2 ,feature3):
    """
    Create a tf.train.Example message ready to written to a file.
    """
    feature = {
        'feature0' : _int64_feature(feature0),
        'feature1' : _int64_feature(feature1),
        'feature2' : _bytes_feature(feature2),
        'feature3' : _float_feature(feature3),
    }
    example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
    return example_proto.SerializeToString()

example_observation = []
serialized_example = serialize_example(False ,4 ,b'goat' ,0.966)
print(serialized_example)
example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto)
tf.data.Dataset.from_tensor_slices(feature1)
features_dataset = tf.data.Dataset.from_tensor_slices((feature0 ,feature1 ,feature2 ,feature3))
print(features_dataset)

def tf_serialize_example(f0,f1,f2,f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),
        tf.string
    )
    return tf.reshape(tf_string, ())

for f0 ,f1 ,f2 ,f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)
    # ?
    # tf_serialize_example(f0 ,f1 ,f2 ,f3)

print(feature0)
print(feature1)
print(feature2)
print(feature3)

serialized_features_dataset = features_dataset.map(tf_serialize_example)
print(serialized_features_dataset)

def generator():
    for feature0 ,feature1 ,feature2 ,feature3 in features_dataset:
        # yield serialize_example(features[0] ,features[1] ,features[2] ,features[3])
        yield serialize_example(feature0 ,feature1 ,feature2 ,feature3)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator,
    output_types = tf.string,
    output_shapes = ()
)
print(serialized_features_dataset)

filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))
feature_description = {
    'feature0':tf.io.FixedLenFeature([],tf.int64 ,default_value = 0),
    'feature1':tf.io.FixedLenFeature([],tf.int64 ,default_value = 0),
    'feature2':tf.io.FixedLenFeature([],tf.string ,default_value = ''),
    'feature3':tf.io.FixedLenFeature([],tf.float32 ,default_value = 0.0),
}
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto ,feature_description)
parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)
for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i],feature1[i],feature2[i],feature3[i])
        writer.write(example)
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)
for raw_record in raw_dataset.take(1):
    example = tf.trian.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

print("Input tensor format data done.")
