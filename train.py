import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import argparse
import glob
from generator import MultiPlotter, preprocess_labels, generator
from model import DilatedCNN

tfe.enable_eager_execution()

parser = argparse.ArgumentParser(description='traing the model')
parser.add_argument('--load', type=bool, default=True, help='setting the learning rate')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='setting the learning rate')
args = parser.parse_args()

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")


def training_cycle(model, gen, memory, learning_rate=args.learning_rate, iterations=200):
  print("Starting training cycle with lr: {} for {} iterations".format(learning_rate, iterations))

  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  for i in range(iterations):
    images, masks, indices = next(gen)
    x = tf.constant(images, dtype=tf.float32)
    y = tf.constant(masks, dtype=tf.float32)
    loss = model.train(x, y, optimizer)
    loss = tf.squeeze(loss)
    memory.update(indices, loss.numpy())

    if i % 10 == 0:
      print("run {} loss: {}".format(i, loss.numpy().mean()))

def display_samples(model, gen):
  images, masks, indices = next(gen)
  x = tf.constant(images, dtype=tf.float32)
  y_hat = model(x)

  plotter = MultiPlotter()
  for index in range(3):
    plotter.append(images[index]) \
      .append(masks[index][:, :, 0]) \
      .append(masks[index][:, :, 1])

    plotter.append(images[index]) \
      .append(y_hat[index][:, :, 0]) \
      .append(y_hat[index][:, :, 1])

def train():
  # load_files()
  batch_size = 12
  from replay_memory import PrioritisedReplayMemory
  memory = PrioritisedReplayMemory(capacity=batch_size * 30)
  gen = generator(x_train, y_train, memory, batch_size=batch_size)
  model = DilatedCNN()

  if args.load:
    model.load()
    # Start with lr=0.001
    training_cycle(model, gen, memory, args.learning_rate, 3000)
    training_cycle(model, gen, memory, args.learning_rate/10, 3000)

  model.save()
  display_samples(model, gen)


def validate():
  model = DilatedCNN()
  model.load()
  batch_size = 12

  mean = tfe.metrics.Mean()

  index_start = 0
  while True:
    index_end = min(x_test.shape[0], index_start+batch_size)
    x = tf.constant(x_test[index_start:index_end], dtype=tf.float32)
    y = tf.constant(y_test[index_start:index_end], dtype=tf.float32)

    y_hat = model(x)
    loss = model.loss(y_hat, y)
    loss = tf.reduce_mean(loss).numpy()
    mean(loss)
    # print("Validation loss is:", loss)
    index_start += batch_size
    if index_start >= x_test.shape[0]:
      break

  print("Validation loss is:", mean.result().numpy())




if __name__ == "__main__":
  train()
  validate()