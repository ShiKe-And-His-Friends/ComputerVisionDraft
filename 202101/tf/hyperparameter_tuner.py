import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

(img_train ,label_train) ,(img_test ,label_test) = keras.datasets.fashion_mnist.load_data()
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28 ,28)))

    # Tune number if unit
    hp_units = hp.Int('units' ,min_value = 32 ,max_value = 512 ,step = 32)
    model.add(keras.layers.Dense(units = hp_units ,activation = 'relu'))
    model.add(keras.layers.Dense(10))

    # Tune optimizer rate
    hp_learning_rate = hp.Choice('learning_rate' ,values = [1e-2 ,1e-3 ,1e-4])
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy']
    )
    return model

tuner = kt.Hyperband(
    model_builder,
    objective = 'val_accuracy',
    max_epochs = 10,
    factor = 3,
    directory = 'my_dir',
    project_name = 'intro_to_kt'
)
stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss' ,patience = 5)
tuner.search(img_train ,label_train ,epochs = 50 ,validation_split = 0.2 ,callbacks = [stop_early])
# get optimal hyperparameter
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
print(f"""
The hyperparameter search is complete.The optimal number of first layer is {best_hps.get('units')} and optimal optimizer rate is {best_hps.get('learning_rate')} .
""")

# train model
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    img_train,
    label_train,
    epochs = 50,
    validation_split = 0.2
)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print("Best epoch: %d" % (best_epoch ,))

# evalute model
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(
    img_train,
    label_train,
    epochs = best_epoch,
    validation_split = 0.2
)
eval_result = hypermodel.evaluate(img_test ,label_test)
print("[test loss ,test accuracy]:" ,eval_result)

print("hyperparameter tune done.")
