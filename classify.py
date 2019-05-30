from __future__ import absolute_import, division, print_function

import util
import numpy as np
import collections
from six.moves import range
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500


def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', 
            input_shape=(784,))])

    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred))

    model.compile(
        loss=loss_fn,
        optimizer=gradient_descent.SGD(learning_rate=0.02),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

nest = tf.contrib.framework.nest

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

""" load data """
emnist_train, emnist_test = util.load_data()

# print(emnist_train.output_types, emnist_train.output_shapes)
# OrderedDict([('label', tf.int32), ('pixels', tf.float32)]) OrderedDict(
# [('label', TensorShape([])), ('pixels', TensorShape([28, 28]))])

# total:3383
NUM_CLIENTS = 80

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

""" construct model """
# iterative_process = tff.learning.build_federated_averaging_process(model_fn)
# state = iterative_process.initialize()
# for round_num in range(11):
#     state, metrics = iterative_process.next(state, federated_train_data)
#     print('round {:2d}, metrics={}'.format(round_num, metrics))


""" customize the model implementation """
MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def create_mnist_variables():
    return MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name='bias',
            trainable=True),
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def mnist_forward_pass(variables, batch):
    y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(flat_labels, 10) * tf.log(y), reduction_indices=[1]))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32))

    num_examples = tf.to_float(tf.size(batch['y']))

    tf.assign_add(variables.num_examples, num_examples)
    tf.assign_add(variables.loss_sum, loss * num_examples)
    tf.assign_add(variables.accuracy_sum, accuracy * num_examples)

    return loss, predictions


def get_local_mnist_metrics(variables):
    return collections.OrderedDict([
        ('weights', variables.weights),
        ('bias', variables.bias),
        ('num_examples', variables.num_examples),
        ('loss', variables.loss_sum / variables.num_examples),
        ('accuracy', variables.accuracy_sum / variables.num_examples)
        ])


@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
    return {
        'num_examples': tff.federated_sum(metrics.num_examples),
        'loss': tff.federated_mean(metrics.loss, metrics.num_examples),
        'accuracy': tff.federated_mean(metrics.accuracy, metrics.num_examples),
        'per_client_num_examples': tff.federated_collect(metrics.num_examples),
        'per_client_loss': tff.federated_collect(metrics.loss),
        'per_client_accuracy': tff.federated_collect(metrics.accuracy),
        'per_client_weights': tff.federated_collect(metrics.weights),
        'per_client_bias': tff.federated_collect(metrics.bias)
    }


class MnistModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                                        tf.float32)),
                                    ('y', tf.TensorSpec([None, 1], tf.int32))])

    # TODO(b/124777499): Remove `autograph=False` when possible.
    @tf.contrib.eager.function(autograph=False)
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        return tff.learning.BatchOutput(loss=loss, predictions=predictions)

    @tf.contrib.eager.function(autograph=False)
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients


class MnistTrainableModel(MnistModel, tff.learning.TrainableModel):

    # TODO(b/124777499): Remove `autograph=False` when possible.
    @tf.contrib.eager.defun(autograph=False)
    def train_on_batch(self, batch):
        output = self.forward_pass(batch)
        optimizer = tf.train.GradientDescentOptimizer(0.02)
        optimizer.minimize(output.loss, var_list=self.trainable_variables)
        return output


iterative_process = tff.learning.build_federated_averaging_process(
    MnistTrainableModel)
state = iterative_process.initialize()

# @test {"timeout": 600, "output": "ignore"}
# state, metrics = iterative_process.next(state, federated_train_data)

w = []
b = []
acc = []
loss = []
num = []
server = {}

server_loss = []
server_acc = []
server_num = []
server_w = []
server_b = []
for i in range(NUM_CLIENTS):
    w.append([])
    b.append([])
    acc.append([])
    loss.append([])
    num.append([])

for round_num in range(18):
    state, metrics = iterative_process.next(state, federated_train_data)
    server_w.append(state.model.trainable.weights)
    server_b.append(state.model.trainable.bias)
    server_loss.append(metrics.loss)
    server_acc.append(metrics.accuracy)
    server_num.append(metrics.num_examples)
    for i in range(NUM_CLIENTS):
        w[i].append(metrics.per_client_weights[i])
        b[i].append(metrics.per_client_bias[i])
        acc[i].append(metrics.per_client_accuracy[i])
        loss[i].append(metrics.per_client_loss[i])
        num[i].append(metrics.per_client_num_examples[i])

    print('round {:2d}'.format(round_num))

""" save results of client training """
result = []
for i in range(NUM_CLIENTS):
    tmp = {}
    tmp['weights'] = w[i]
    tmp['bias'] = b[i]
    tmp['accuracy'] = acc[i]
    tmp['loss'] = loss[i]
    tmp['num'] = num[i]
    result.append(tmp)

result_file = './results/clients_data.json'
util.save_as_json(result, result_file)

""" save server training data """
server['weights'] = server_w
server['bias'] = server_b
server['loss'] = server_loss
server['accuracy'] = server_acc
server['num'] = server_num

server_file = './results/server_data.json'
util.save_as_json(server, server_file)
