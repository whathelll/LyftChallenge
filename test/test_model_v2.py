import unittest
import cv2
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model_v2 import DownModule, DilatedCNN, ConvModule, DilatedModule, UpSampleModule
import numpy as np
import glob
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tfe.enable_eager_execution()

class TestModel(unittest.TestCase):
  def setUp(self):
    pass

  def test_down_sample(self):
    model = DownModule(filters=13)
    batch = tf.random_normal((1, 600, 800, 3))
    out = model(batch)
    self.assertEqual(out.shape, (1, 300, 400, 16))

  def test_conv_module(self):
    model = ConvModule(reduce_depth=16, expand_depth=32, dropout_rate=0.1)
    batch = tf.random_normal((1, 300, 400, 32))
    out = model(batch)
    self.assertEqual(out.shape, (1, 300, 400, 32))


  def test_dilated_module(self):
    model = DilatedModule(reduce_depth=8, expand_depth=32, dropout_rate=0.1, dilation_rate=(4, 4))
    batch = tf.random_normal((1, 300, 400, 32))
    out = model(batch)
    self.assertEqual(out.shape, (1, 300, 400, 32))

  def test_upsample_module(self):
    model = UpSampleModule(expand_depth=32, dropout_rate=0.1)
    batch = tf.random_normal((1, 150, 200, 32))
    out = model(batch)
    self.assertEqual(out.shape, (1, 300, 400, 32))



  def xtest_model(self):
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
    # print("Y", y)
    # print("y shape:", y.shape)
    out = model.loss(y_hat, y)
    #
    # print("out", out)
    # print(out.numpy().mean())






if __name__ == "__main__":
  unittest.main()