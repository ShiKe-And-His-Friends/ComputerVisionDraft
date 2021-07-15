import json
import os
import sys
import tensorflow as tf
import distribute_multi_mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG' ,None)

if '.' not in sys.path:
    sys.path.insert(0 ,'.')
batch_size = 64
single_worker_dataset = distribute_multi_mnist.mnist_dataset(batch_size)
single_worker_model = distribute_multi_mnist.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset ,epochs = 3 ,steps_per_epoch = 70)

tf_config = {
    'cluster':{
        'worker':['localhost:12345' ,'localhost:23456']
    },
    'task':{'type':'worker' ,'index':0}
}
print(json.dumps(tf_config))
os.environ['GREETINGS'] = 'Hello Tensorflow'

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    multi_worker_model = distribute_multi_mnist.build_and_compile_cnn_model()


print("Distribute multi keras done.")
