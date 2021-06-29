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
print("\n\n\n 0:\n")
print(abalone_features)
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

# predict tatanic survivy
titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
print(titanic.head())
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop("survived")
input = tf.keras.Input(shape =() ,dtype = tf.float32)
result = 2 * input + 1
print(result)
calc = tf.keras.Model(inputs = input ,outputs = result)
print(calc(1).numpy())
print(calc(2).numpy())

inputs = {}
for name ,column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape = (1 ,) ,name = name ,dtype = dtype)
print(inputs)
numeric_inputs = {name:input for name ,input in inputs.items()
    if input.dtype == tf.float32}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()

print("\n\n\n 1:\n")
y1 = numeric_inputs.keys()
print(y1)
y2 = titanic[y1]
print("\n\n\n 2:\n")
print(y2)
y3 = np.array(y2)
print("\n\n\n 3:\n")
print(y3)
print("\n\n\n 4:\n")
# y4 = tf.cast(y3 ,dtype = tf.float32)
y4 = y3.astype(np.float32)
print(y4)
norm.adapt(y3)

# norm.adapt(np.array(titanic[numeric_inputs.keys()]))
# norm.adapt(titanic_features)
all_numeric_inputs = norm(x)
print(all_numeric_inputs)
preprocessed_inputs = [all_numeric_inputs]

print("\nInput CVS data done.\n")

