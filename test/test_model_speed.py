import unittest
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model import DilatedCNN
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tfe.enable_eager_execution()

class TestModel(unittest.TestCase):
  def setUp(self):
    # self.downsample = DownSample()
    pass


  def test_model(self):
    model = DilatedCNN()
    # init weights
    x = tf.random_normal((1, 320, 400, 3))
    out = model(x)
    print(out)
    print(out.shape)

    x = tf.random_normal((1, 320, 400, 3))
    start = time.time()
    for i in range(10):
      test = model(x)

    duration = (time.time() - start) / 10
    print("average duration:", round(duration, 4))






if __name__ == "__main__":
  unittest.main()