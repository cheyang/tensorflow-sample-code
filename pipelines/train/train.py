import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
#from tf.data import base

# import tensorflow_datasets as tfds

#from tensorflow.data import base

IRIS_TRAIN='iris_training.csv'
IRIS_TEST='iris_test.csv'

base.load_csv_with_header(filename=IRIS_TRAIN, features_dtype=np.float32, target_dtype=np.int)
train_set = base.load_csv_with_header(filename=IRIS_TRAIN,features_dtype=np.float32,
                                         target_dtype=np.int)

test_set = base.load_csv_with_header(filename=IRIS_TEST, features_dtype=np.float32, target_dtype=np.int)