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
    features = {
        'feature0' : _int64_feature(feature0),
        'feature1' : _int64_feature(feature1),
        'feature2' : _bytes_feature(feature2),
        'feature3' : _float_feature(feature3),
    }
    example_proto = tf.train.Example(features = tf.train.Features(feature = features))
    return example_proto.SerializeToString()
example_observation = []
serialized_example = serialize_example(False ,4 ,b'goat' ,0.9876)
print(serialized_example)
example_proto = tf.train.Example.FromString(serialize_example)
print(example_proto)
tf.data.Dataset.from_tensor_slices(feature1)
features_dataset = tf.data.Dataset.from_tensor_slices((feature0 ,feature1 ,feature2 ,feature3))
print(features_dataset)
for f0 ,f1 ,f2 ,f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)

def tf_serialize_example(f0 ,f1 ,f2 ,f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0 ,f1 ,f2 ,f3),
        tf.string)
    )
    return tf.reshape(tf_string ,())
tf_serialize_example(f0 ,f1 ,f2 ,f3)


print("Input tensor format data done.")
