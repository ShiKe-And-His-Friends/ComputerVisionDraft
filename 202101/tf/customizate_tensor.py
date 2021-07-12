import tensorflow as tf
import numpy as np
import time
import tempfile

print(tf.add(1 ,2))
print(tf.add([1 ,2] ,[3 ,4]))
print(tf.square(5))
print(tf.reduce_sum([1 ,2 ,3]))
print(tf.square(2) + tf.square(3))
x = tf.matmul([[1]] ,[[2 ,3]])
print(x)
print(x.shape)
print(x.dtype)

ndarray = np.ones([3 ,3])
print("TensorFlow operations convert numpy arrays to Tensors automatically.")
tensor = tf.multiply(ndarray ,42)
print(tensor)
print("And Numpy opertations convert Tensors to numpy arrays automatically.")
print(np.add(tensor ,1))
print("The .numpy() methods explicitly convert a Tensor to a numpy array.")
print(tensor.numpy())
x = tf.random.uniform([3 ,3])
print("Is there a GPU availabel:"),
print(tf.config.list_physical_devices("GPU"))
print("Is the Tensor on CPU #0: "),
print(x.device.endswith('GPU:0'))
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x ,x)
    result = time.time() - start
    print("10 loops:{:0.2f}ms".format(1000 * result))
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000 ,1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

if tf.config.list_physical_devices("GPU"):
    print("On GPU:")
    with tf.device("GPU:0"):
        x = tf.random.uniform([1000 ,1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
'''
Tensorflow GPU
https://blog.csdn.net/wbing96/article/details/108914071
On CPU:
10 loops:55.00ms
On GPU:
10 loops:456.97ms
'''

ds_tensors = tf.data.Dataset.from_tensor_slices([1 ,2 ,3 ,4 ,5 ,6])
_, filename = tempfile.mkstemp()
with open(filename ,"w") as f:
    f.write(
        """
        Line 1
        Line 2
        Line 3
        """
    )
ds_file = tf.data.TextLineDataset(filename)
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)
for x in ds_file:
    print(x)

print("Customizate tensor done.")
