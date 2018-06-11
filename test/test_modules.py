import unittest

from modules import ConvModule, SqueezeNetFireModule, ConvPoolModule
from model_v4 import DilatedCNN
from UNet import UNet
from loss import Loss
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def model_perf(model, x):
  inputs = tf.placeholder(tf.float32, [None, 300, 400, 3])
  results = model(inputs)

  iter = 100

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    now = time.time()
    for i in range(iter):
      y = sess.run(results, feed_dict={inputs: x})
  duration = (time.time() - now) / iter
  return duration

def model_params(model):
  size = 0
  for v in model.variables:
    size += tf.size(v)
  return size.numpy()

class TestModules(unittest.TestCase):
  def setUp(self):
    self.x = np.random.normal(size=(1, 300, 400, 3)) #tf.random_normal((1, 300, 400, 3))


  @unittest.skip("skipping")
  def test_conv_module(self):
    model = ConvModule(64, kernel_size=(3, 3))
    print("ConvModule (3,3)x64 Average Time: {:.3f}".format(model_perf(model, self.x)))

    model = ConvModule(64, kernel_size=(1, 1))
    print("ConvModule (1,1)x64 Average Time: {:.3f}".format(model_perf(model, self.x)))

    model = ConvModule(128, kernel_size=(3, 3))
    print("ConvModule (3,3)x128 Average Time: {:.3f}".format(model_perf(model, self.x)))

    model = ConvModule(128, kernel_size=(1, 1))
    print("ConvModule (1,1)x128 Average Time: {:.3f}".format(model_perf(model, self.x)))

# ConvModule (3,3)x64 Average Time: 0.079
# ConvModule (1,1)x64 Average Time: 0.073
# ConvModule (3,3)x128 Average Time: 0.151
# ConvModule (1,1)x128 Average Time: 0.149

  @unittest.skip("skipping")
  def test_SqueezeNetFire_module(self):
    model = SqueezeNetFireModule(16, 64)
    print("SqueezeNetFireModule 16x64 Average Time: {:.3f}".format(model_perf(model, self.x)))

    model = SqueezeNetFireModule(32, 128)
    print("SqueezeNetFireModule 32x128 Average Time: {:.3f}".format(model_perf(model, self.x)))

# SqueezeNetFireModule 16x64 Average Time: 0.050
# SqueezeNetFireModule 32x128 Average Time: 0.113

  @unittest.skip("skipping")
  def test_ConvPool_module(self):
    model = ConvPoolModule(64, conv_layers=2)
    print("ConvPoolModule 64x2 Average Time: {:.3f}".format(model_perf(model, self.x)))

    model = ConvPoolModule(128, conv_layers=4)
    print("ConvPoolModule 128x4 Average Time: {:.3f}".format(model_perf(model, self.x)))

    y = model.predict(self.x)
    self.assertEqual((1, 150, 200, 128), y.shape)


  """This turned out to be the same performance as using Keras layers"""
  @unittest.skip("skipping")
  def test_normal_conv_module(self):
    inputs = tf.placeholder(tf.float32, [None, 300, 400, 3])

    conv = tf.layers.conv2d(inputs, 64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    bn = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(bn)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      now = time.time()
      for i in range(100):
        y = sess.run(relu, feed_dict={inputs: self.x})

      duration = (time.time() - now) / 100
      print("Time taken on pure tensorflow:", duration)

  @unittest.skip("skipping")
  def test_keras_model(self):
    inputs = tf.keras.layers.Input(shape=(300, 400, 3))
    out = ConvModule(64, kernel_size=(3, 3))(inputs)
    # out = ConvModule(64, kernel_size=(3, 3))(out)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer, loss='mean_squared_error')
    # model.summary()

    iter = 100
    now = time.time()
    for i in range(iter):
      y = model.predict_on_batch(self.x)
    duration = (time.time() - now) / iter
    print("Time taken on keras model:", duration)


  @unittest.skip("skipping")
  def test_keras_model_v4_model(self):
    x = np.random.normal(size=(1, 320, 400, 3))
    inputs = tf.keras.layers.Input(shape=(320, 400, 3))
    out = DilatedCNN()(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer, loss='mean_squared_error')
    # model.summary()
    iter = 100
    now = time.time()
    for i in range(iter):
      y = model.predict(x)
    duration = (time.time() - now) / iter
    print("Time taken on keras model:", duration)





if __name__ == '__main__':
    unittest.main()