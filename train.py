import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import argparse
import glob
from generator import MultiPlotter, preprocess_labels, generator
from model import DilatedCNN

tfe.enable_eager_execution()

parser = argparse.ArgumentParser(description='traing the model')
parser.add_argument('--load', type=bool, default=False, help='whether to load the model')
parser.add_argument('--save', type=bool, default=False, help='whether to save the model')
parser.add_argument('--train_once', type=bool, default=False, help='train only once, used with iterations')
parser.add_argument('--train_once_iterations', type=int, default=1000, help='train only once, used with iterations')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='setting the learning rate')
args = parser.parse_args()

x_train = np.load("x_train_s.npy")
x_test = np.load("x_test_s.npy")
y_train = np.load("y_train_s.npy")
y_test = np.load("y_test_s.npy")

def training_cycle(model, gen, memory, learning_rate=args.learning_rate, iterations=200):
  print("Starting training cycle with lr: {} for {} iterations".format(learning_rate, iterations))
  now = time.time()
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=1e-4)
  metric_mean = tfe.metrics.Mean()

  for i in range(iterations):
    # images, masks, indices = next(gen)
    images, masks = next(gen)
    x = tf.constant(images, dtype=tf.float32)
    y = tf.constant(masks, dtype=tf.float32)
    # x = tf.image.rgb_to_grayscale(x)
    # x = tf.image.resize_images(x, (320, 400))
    # y = tf.image.resize_images(y, (320, 400))

    loss = model.train(x, y, optimizer)
    loss = tf.squeeze(loss)
    # memory.update(indices, loss.numpy())
    metric_mean(loss)
    if i % 50 == 0:
      print("run {} loss: {}".format(i, metric_mean.result()))
      metric_mean = tfe.metrics.Mean()
  duration = time.time() - now
  print("Training cycle took:", duration/60)
  validate(model)

def display_samples(model, gen):
  images, masks = next(gen)
  x = tf.constant(images, dtype=tf.float32)
  y_hat = model(x)

  plotter = MultiPlotter(columns=5)
  for index in range(5):
    plotter.append(images[index]) \
      .append(masks[index][:, :, 0]) \
      .append(masks[index][:, :, 1]) \
      .append(y_hat[index][:, :, 0]) \
      .append(y_hat[index][:, :, 1])
  plotter.show()

def train():
  # load_files()
  batch_size = 12
  from replay_memory import PrioritisedReplayMemory
  memory = PrioritisedReplayMemory(capacity=batch_size)
  gen = generator(x_train, y_train, memory, batch_size=batch_size)
  model = DilatedCNN()

  if args.load:
    print("loading model")
    model.load()

  training_cycle(model, gen, memory, args.learning_rate, 2000)
  training_cycle(model, gen, memory, args.learning_rate/10, 2000)
  training_cycle(model, gen, memory, args.learning_rate/100, 2000)
  training_cycle(model, gen, memory, args.learning_rate/1000, 2000)
  training_cycle(model, gen, memory, args.learning_rate/10000, 1000)
  training_cycle(model, gen, memory, args.learning_rate/10000, 1000)
  #
  # training_cycle(model, gen, memory, args.learning_rate/100000, 1500)
  # training_cycle(model, gen, memory, args.learning_rate/100000, 1500)
  # training_cycle(model, gen, memory, args.learning_rate/1000000, 1500)
  model.save()

  # display_samples(model, gen)
  # display_samples(model, gen)
  # display_samples(model, gen)


def validate(model=None):
  if model is None:
    model = DilatedCNN()
    model.load()
  batch_size = 12

  mean = tfe.metrics.Mean()

  index_start = 0
  while True:
    index_end = min(x_test.shape[0], index_start+batch_size)
    x = tf.constant(x_test[index_start:index_end], dtype=tf.float32)
    y = tf.constant(y_test[index_start:index_end], dtype=tf.float32)

    # x = tf.image.resize_images(x, (160, 200))
    # y = tf.image.resize_images(y, (160, 200))

    y_hat = model(x).numpy()
    y_hat = np.where(y_hat >= 0.5, 1, 0).astype("float32")

    loss = model.loss(y_hat, y)
    mean(tf.reduce_mean(loss))
    # print("Validation loss is:", loss)
    index_start += batch_size
    if index_start >= x_test.shape[0]:
      break

  print("Validation loss is:", mean.result().numpy())




if __name__ == "__main__":
  if args.train_once:
    print("training once")
    batch_size = 12
    from replay_memory import PrioritisedReplayMemory
    memory = PrioritisedReplayMemory(capacity=batch_size)
    gen = generator(x_train, y_train, memory, batch_size=batch_size)
    model = DilatedCNN()
    if args.load:
      model.load()

    training_cycle(model, gen, memory, args.learning_rate, args.train_once_iterations)

    if args.save:
      model.save()

    display_samples(model, gen)

  else:
    train()
  # validate()