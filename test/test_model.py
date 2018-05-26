import unittest
import cv2
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model import DownSample, DilationLayer, UpSample, DilatedCNN
import numpy as np
import glob
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tfe.enable_eager_execution()

class TestModel(unittest.TestCase):
  def setUp(self):
    # self.downsample = DownSample()
    pass

  # def test_down_sample(self):
  #   model = DownSample(filters=13)
  #   batch = tf.random_normal((1, 600, 800, 3))
  #   out = model(batch)
  #   self.assertEqual(out.shape, (1, 300, 400, 16))
  #
  #
  #
  # def test_dilation_layer(self):
  #   model = DilationLayer(filters=13)
  #   batch = tf.random_normal((1, 75, 100, 25))
  #   out = model(batch)
  #
  #   self.assertEqual(out.shape, (1, 75, 100, 13))
  #
  #
  # def test_dilation_layer(self):
  #   model = UpSample(filters=13)
  #   batch = tf.random_normal((1, 75, 100, 25))
  #   out = model(batch)
  #   self.assertEqual(out.shape, (1, 150, 200, 13))


  def test_model(self):
    model = DilatedCNN()
    # x = tf.random_normal((1, 600, 800, 3))
    # print(x.shape)
    #
    # test = model(x)
    # print(test.shape)
    y_hat = np.array([[[1, 0, 1, 1],
                       [1, 0, 1, 0]],
                      [[1, 0, 0, 1],
                       [0, 0, 1, 0]]
                      ])

    y = np.array([[[1, 0, 1, 0],
                   [1, 0, 1, 1]],
                  [[1, 1, 0, 1],
                   [1, 0, 0, 1]]
                  ])
    print("Y", y)
    # y_hat = np.expand_dims(y_hat, axis=1)
    # y = np.expand_dims(y, axis=1)
    print("y shape:", y.shape)
    out = model.loss(y_hat, y)

    print("out", out)
    print(out.numpy().mean())






if __name__ == "__main__":
  unittest.main()