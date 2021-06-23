import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

print(tf.__version__)

# download baseline
logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir ,ignore_errors = True)

# prepare data
gz = tf.keras.utils.get_file('HIGGS.csv.gz' ,'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
ds = tf.data.experimental.CsvDataset(gz ,[float(),] * (FEATURES + 1) ,compression_type = "GZIP")

def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:] ,1)
    return features ,label

packed_ds = ds.batch(10000).map(pack_row).unbatch()
for features ,label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten() ,bins = 101)

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN //BATCH_SIZE
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
print(train_ds)
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# demonstrate overfit
# methods.1 schedule produre
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps = STEPS_PER_EPOCH * 1000,
        decay_rate = 1,
        staircase = False)
def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)
step = np.linspace(0 ,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8 ,6))
plt.plot(step / STEPS_PER_EPOCH ,lr)
plt.ylim([0 ,max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()

def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy' ,patience = 200),
        tf.keras.callbacks.TensorBoard(logdir / name),
    ]

def compile_and_fit(model ,name ,optimizer = None ,max_epochs = 1000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer = optimizer,
            loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
            metrics = [
                tf.keras.losses.BinaryCrossentropy(
                    from_logits = True ,name = 'binary_crossentropy'),
                'accuracy'])
    print(model.summary())

    history = model.fit(train_ds,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = max_epochs,
            validation_data = validate_ds,
            callbacks = get_callbacks(name),
            verbose = 0)
    return history
# test1. tiny model
size_histories = {}

tiny_model = tf.keras.Sequential([
    layers.Dense(16 ,activation = 'elu',
        input_shape = (FEATURES,)),
    layers.Dense(1)
])
size_histories['Tiny'] = compile_and_fit(tiny_model ,'sizes/Tiny')
# test2. small model
small_model = tf.keras.Sequential([
    layers.Dense(16 ,activation = 'elu' ,input_shape = (FEATURES,)),
    layers.Dense(16 ,activation = 'elu'),
    layers.Dense(1)
])
size_histories['Small'] = compile_and_fit(small_model ,'sizes/Small')
# test3. medium model
medium_model = tf.keras.Sequential([
    layers.Dense(64 ,activation = 'elu' ,input_shape = (FEATURES,)),
    layers.Dense(64 ,activation = 'elu'),
    layers.Dense(64 ,activation = 'elu'),
    layers.Dense(1)
])
size_histories['Medium'] = compile_and_fit(medium_model ,"sizes/Medium")
# test4. large model
large_model = tf.keras.Sequential([
    layers.Dense(512 ,activation = 'elu',input_shape = (FEATURES,)),
    layers.Dense(512 ,activation = 'elu'),
    layers.Dense(512 ,activation = 'elu'),
    layers.Dense(512 ,activation = 'elu'),
    layers.Dense(1)
])
size_histories['large'] = compile_and_fit(large_model ,"sizes/large")

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy' ,smoothing_std = 10)
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5 ,max(plt.xlim())])
plt.ylim([0.5 ,0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()
# draw 

# strategies 

# regularize

# dropout 

# draw

print("\nRegularize overfit done.\n")
