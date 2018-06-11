import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import argparse
import glob
from generator import MultiPlotter, preprocess_labels, generator
from replay_memory import PrioritisedReplayMemory
from model_v4 import DilatedCNN

parser = argparse.ArgumentParser(description='traing the model')
parser.add_argument('--load', type=bool, default=False, help='whether to load the model')
parser.add_argument('--save', type=bool, default=False, help='whether to save the model')
parser.add_argument('--train_once', type=bool, default=False, help='train only once, used with iterations')
parser.add_argument('--display_sample', type=bool, default=False, help='train only once, used with iterations')
parser.add_argument('--train_once_iterations', type=int, default=1000, help='train only once, used with iterations')
parser.add_argument('--learning_rate', type=float, default=0.01, help='setting the learning rate')
args = parser.parse_args()

x_train = np.load("x_train_s.npy")
x_test = np.load("x_test_s.npy")
y_train = np.load("y_train_s.npy")
y_test = np.load("y_test_s.npy")

def training_cycle(model, gen, memory, learning_rate=args.learning_rate, iterations=200):
  print("Starting training cycle with lr: {} for {} iterations".format(learning_rate, iterations))
  now = time.time()
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)


  x = tf.placeholder(tf.float32, [None, 320, 400, 3])
  y = tf.placeholder(tf.float32, [None, 320, 400, 3])
  y_hat = model(x)
  loss, car_loss, road_loss = model.loss(y_hat, y)
  loss_reduced_mean = tf.reduce_mean(loss)
  train_op = optimizer.minimize(loss_reduced_mean)

  loss_mean, loss_mean_update_op = tf.metrics.mean(tf.squeeze(loss))
  car_loss_mean, car_loss_mean_update_op = tf.metrics.mean(tf.squeeze(car_loss))
  road_loss_mean, road_loss_mean_update_op = tf.metrics.mean(tf.squeeze(road_loss))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(iterations):
      images, masks, indices = next(gen)

      sess.run([train_op, loss_mean_update_op, car_loss_mean_update_op, road_loss_mean_update_op], feed_dict={x: images, y: masks})
      # memory.update(indices, car_loss.numpy())

      if (i+1) % 50 == 0 or i == 0:
        print("run {} loss: {}, car loss: {}, road loss:{}".format(i + 1, sess.run(loss_mean), sess.run(car_loss_mean),
                                                                   sess.run(road_loss_mean)))
    duration = time.time() - now
    print("Training cycle took:", duration/60)
    model.save()
  validate(model)  #this fails at the moment. Saving doesn't quite work.

def display_samples(model, gen):
  images, masks, indices = next(gen)
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
  memory = PrioritisedReplayMemory(capacity=batch_size*100, e=0.01)
  gen = generator(x_train, y_train, memory, batch_size=batch_size)
  model = DilatedCNN()

  if args.load:
    print("loading model")
    model.load()

  training_cycle(model, gen, memory, 1e-2, 10)
  model.save()

  # for i in range(7):
  #   training_cycle(model, gen, memory, 1e-3, 1000)
  #   model.save()
  #
  # for i in range(5):
  #   training_cycle(model, gen, memory, 1e-4, 1000)
  #   model.save()
  #
  # for i in range(5):
  #   training_cycle(model, gen, memory, 5e-5, 1000)
  #   model.save()
  #
  # for i in range(5):
  #   training_cycle(model, gen, memory, 1e-5, 1000)
  #   model.save()


def validate(model=None):

  # model.load()
  batch_size = 12


  x = tf.placeholder(tf.float32, [None, 320, 400, 3])
  y = tf.placeholder(tf.float32, [None, 320, 400, 3])
  y_hat = model(x)
  loss, car_loss, road_loss = model.loss(y_hat, y)

  loss_mean, loss_mean_update_op = tf.metrics.mean(tf.squeeze(loss))
  now = time.time()
  index_start = 0
  with tf.Session() as sess:
    # model.load()
    while True:
      index_end = min(x_test.shape[0], index_start+batch_size)
      images = x_test[index_start:index_end]
      masks = y_test[index_start:index_end]

      sess.run([y_hat, loss_mean_update_op], feed_dict={x: images, y: masks})

      index_start += batch_size
      if index_start >= x_test.shape[0]:
        break

    duration = time.time() - now
    print("Validation took:", duration/60)
    print("Validation loss is:", sess.run(loss_mean))

def init():
  batch_size = 12
  memory = PrioritisedReplayMemory(capacity=batch_size, e=0.01)
  gen = generator(x_train, y_train, memory, batch_size=batch_size)
  model = DilatedCNN()
  return model, gen, memory


if __name__ == "__main__":
  if args.display_sample:
    print("display_samples")
    model, gen, memory = init()
    model.load()
    display_samples(model, gen)
    display_samples(model, gen)
    display_samples(model, gen)
    display_samples(model, gen)
    display_samples(model, gen)

  elif args.train_once:
    print("training once")
    model, gen, memory = init()
    if args.load:
      model.load()

    training_cycle(model, gen, memory, args.learning_rate, args.train_once_iterations)

    if args.save:
      model.save()

    display_samples(model, gen)

  else:
    train()
  # validate()