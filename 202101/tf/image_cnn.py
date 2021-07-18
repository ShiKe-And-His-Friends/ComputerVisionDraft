import tensorflow as tf
from tensorflow.keras import datasets ,layers ,models
import matplotlib.pyplot as plt

(train_images ,train_labels),(test_images ,test_labels) = datasets.cifar10.load_data()
