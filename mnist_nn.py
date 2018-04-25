from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pandas as pd
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/")

train = pd.DataFrame(data = mnist.train.images)
labels = pd.DataFrame(data = mnist.train.labels).astype(np.int32)

test = pd.DataFrame(data = mnist.test.images)
test_labels = pd.DataFrame(data = mnist.test.labels).astype(np.int32)

metrics = train_NN(
  learning_rate = 0.1,
  hidden_units = [200, 50],
  batch_size = 100,
  steps = 5000
)

def train_NN(
  learning_rate,
  hidden_units,
  batch_size,
  steps
):
  nn = tf.estimator.DNNClassifier(
    feature_columns = create_feature_cols(),
    hidden_units = hidden_units,
    n_classes = 10,
    optimizer = tf.train.AdagradOptimizer(
      learning_rate = learning_rate
    )
  )

  nn.train(
    input_fn = lambda: input_fn(train, labels, batch_size=batch_size),
    steps = steps
  )

  metrics = nn.evaluate(
    input_fn = lambda: input_fn(test, test_labels, predict=True)
  )

  return metrics

def create_feature_cols():
  pixels = tf.feature_column.numeric_column(key="pixels", shape=784)
  return set([pixels])

def input_fn(features, labels, batch_size=1, predict=False):
    dataset = tf.data.Dataset.from_tensor_slices(({"pixels": features}, labels))
    if predict:
      dataset = dataset.repeat(1).batch(1)
    else:
      dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset
