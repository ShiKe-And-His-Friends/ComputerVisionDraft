import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

np.set_printoptions(precision = 3 ,suppress = True)

# download data
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names = ["Length" ,"Diameter" ,"Height" ,"Whole weight" ,"Shucked weight"
        ,"Viscera weight" ,"Shell weight" ,"Age"]
)
print(abalone_train.head())
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')
abalone_features = np.array(abalone_features)
print(abalone_labels)
print(abalone_features)

# simple train
abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])
abalone_model.compile(
    loss = tf.losses.MeanSquaredError(),
    optimizer = tf.optimizers.Adam()
)
abalone_model.fit(
    abalone_features,
    abalone_labels,
    epochs = 10
)
normalize = preprocessing.Normalization()
normalize.adapt(abalone_features)
norm_abalone_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
])
norm_abalone_model.compile(
    loss = tf.losses.MeanSquaredError(),
    optimizer = tf.optimizers.Adam()
)
norm_abalone_model.fit(
    abalone_features,
    abalone_labels,
    epochs = 10
)

print("\nInput CVS data done.\n")

