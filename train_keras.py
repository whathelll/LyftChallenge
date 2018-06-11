import cv2
import time
import numpy as np
import tensorflow as tf
import argparse
from generator import MultiPlotter, generator, test_generator
from replay_memory import PrioritisedReplayMemory
from model_v4 import DilatedCNN as SegModel
from UNet import UNet as SegModel
model_save_file = "model_v4.h5"
model_save_file = "UNet.h5"

parser = argparse.ArgumentParser(description='traing the model')
parser.add_argument('--load', type=bool, default=False, help='whether to load the model')
parser.add_argument('--save', type=bool, default=False, help='whether to save the model')
parser.add_argument('--validate', type=bool, default=False, help='whether to save the model')
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
  tf.keras.backend.set_value(model.optimizer.lr, learning_rate)

  for i in range(iterations):
    images, masks = next(gen)
    model.fit(images, masks, batch_size=2, verbose=1)

  duration = time.time() - now
  print("Training cycle took:", duration/60)
  validate(model)

def display_samples(model, gen):
  images, masks = next(gen)
  x = images
  y_hat = model.predict_on_batch(x)

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
  batch_size = 12*100
  from replay_memory import PrioritisedReplayMemory
  memory = PrioritisedReplayMemory(capacity=batch_size*10, e=0.01)
  gen = generator(x_train, y_train, memory, batch_size=batch_size)
  # gen = test_generator(x_train[:5], y_train[:5], batch_size=batch_size)
  model = SegModel.build_keras_model()

  if args.load:
    print("loading model")
    model.load_weights(model_save_file)

  training_cycle(model, gen, memory, 1e-5, 5)
  model.save_weights(model_save_file)

  for i in range(5):
    training_cycle(model, gen, memory, 1e-5, 10)
    model.save_weights(model_save_file)

  for i in range(3):
    training_cycle(model, gen, memory, 1e-6, 10)
    model.save_weights(model_save_file)

  for i in range(2):
    training_cycle(model, gen, memory, 1e-7, 10)
    model.save_weights(model_save_file)

def validate(model=None):
  if model is None:
    # model = SegModel.build_keras_model()
    model = model.load_weights(model_save_file)

  now = time.time()

  losses = model.evaluate(x_test, y_test, batch_size=12)
  print("loss: {:.4f} - Lyft_FScore: {:.4f} - Lyft_car_Fscore: {:.4f} - Lyft_road_Fscore: {:.4f}".format(losses[0], losses[1], losses[2], losses[3]))

  duration = time.time() - now
  print("Validation took:", duration/60)
  # print("Validation loss is:", np.mean(losses))

def init():
  batch_size = 12
  memory = PrioritisedReplayMemory(capacity=batch_size, e=0.01)
  gen = generator(x_train, y_train, memory, batch_size=batch_size)
  model = SegModel.build_keras_model()
  return model, gen, memory


if __name__ == "__main__":
  if args.display_sample:
    print("display_samples")
    model, gen, memory = init()
    model.load_weights(model_save_file)
    display_samples(model, gen)
    display_samples(model, gen)
    display_samples(model, gen)
    display_samples(model, gen)
    display_samples(model, gen)

  elif args.validate:
    model, gen, memory = init()
    model.load_weights(model_save_file)
    validate(model)
  elif args.train_once:
    print("training once")
    model, gen, memory = init()
    if args.load:
      model.load_weights(model_save_file)

    training_cycle(model, gen, memory, args.learning_rate, args.train_once_iterations)

    if args.save:
      model.save_weights(model_save_file)

    display_samples(model, gen)

  else:
    train()
  # validate()