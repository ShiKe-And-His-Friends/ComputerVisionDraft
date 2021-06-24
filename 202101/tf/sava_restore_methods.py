import os
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

(train_images ,train_labels) ,(test_images ,test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1 ,28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1 ,28 * 28) / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512 ,activation = 'relu' ,input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer = 'adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [tf.metrics.SparseCategoricalAccuracy()]
    )
    return model
model = create_model()
print(model.summary())

# checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    save_weights_only = True,
    verbose = 1
)
model.fit(
    train_images,
    train_labels,
    epochs = 10,
    validation_data = (test_images ,test_labels),
    callbacks = [cp_callback]
)
text1 = os.listdir(checkpoint_dir)
print(text1)

model = create_model()
loss ,acc = model.evaluate(
    test_images,
    test_labels,
    verbose = 2
)
print("Untrained model ,accuracy: {:5.2f}%".format(100 * acc))
model.load_weights(checkpoint_path)
loss ,acc = model.evaluate(
    test_images,
    test_labels,
    verbose = 2
)
print("Restored model ,accuracy: {:5.2f}%".format(100 * acc))

print("Save and restore model done.")
