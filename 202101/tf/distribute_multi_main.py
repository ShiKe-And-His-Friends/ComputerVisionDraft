import os
import json
import tensorflow as tf
import distribute_multi_mnist

per_worker_batch_size = 64
tf_config = json.load(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distibute.MultiWorkerMirroredStrategy()
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = distribute_multi_mnist.mnist_dataset(global_batch_size)
with strategy.scope():
    multi_worker_model = distribute_multi_mnist.build_and_compile_cnn_model()
multi_worker_model.fit(
    multi_worker_dataset,
    epochs = 3,
    steps_per_epoch = 70
)
