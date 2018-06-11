import unittest
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model_v4 import DilatedCNN
from UNet import UNet
import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tfe.enable_eager_execution()

class TestModel(unittest.TestCase):
  def setUp(self):
    # self.downsample = DownSample()
    pass


  def test_model(self):
    model = DilatedCNN()
    # init weights
    x = tf.random_normal((1, 600, 800, 3))
    x = tf.image.resize_images(x, (320, 400))
    out = model(x)
    print(out.shape)

    x = tf.random_normal((1, 600, 800, 3))
    x = tf.image.resize_images(x, (320, 400))
    start = time.time()
    steps = 100
    for i in range(steps):
      test = model(x)

    duration = (time.time() - start) / steps
    print("average duration:", round(duration, 4))






if __name__ == "__main__":
  unittest.main()