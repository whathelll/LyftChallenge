import unittest

from modules import ConvModule, SqueezeNetFireModule
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tfe.enable_eager_execution()

"""recorded stats are run on CPU"""
def model_perf(model, x):
  iter = 100
  now = time.time()
  for i in range(iter):
    model(x)
  duration = (time.time() - now) / iter
  return duration


def model_train_perf(model, x, y):
  iter = 100
  optimizer = tf.train.AdamOptimizer()
  now = time.time()
  for i in range(iter):
    with tfe.GradientTape() as tape:
      y_hat = model(x)
      loss = (y - y_hat) ** 2
      mean_loss = tf.reduce_mean(loss)
    grad = tape.gradient(mean_loss, model.variables)
    optimizer.apply_gradients(zip(grad, model.variables), global_step=tf.train.get_or_create_global_step())
  duration = (time.time() - now) / iter


  def train(self, x, y, optimizer):
    grads, loss_value, car_loss, road_loss = self.grad(x, y)

    return loss_value, car_loss, road_loss
  return duration

def model_params(model):
  size = 0
  for v in model.variables:
    size += tf.size(v)
  return size.numpy()

class TestModules(unittest.TestCase):
  def setUp(self):
    self.x = tf.random_normal((1, 300, 400, 3))

  @unittest.skip("")
  def test_conv_module(self):
    model = ConvModule(64, kernel_size=(3, 3))
    self.y = tf.random_normal((1, 300, 400, 64))
    print("ConvModule (3,3)x64 Average Time: {:.3f}".format(model_perf(model, self.x)))
    print("ConvModule (3,3)x64 Average Train Time: {:.3f}".format(model_train_perf(model, self.x, self.y)))
    print("ConvModule (3,3)x64 Parameters: {:,d}".format(model_params(model)))

    model = ConvModule(64, kernel_size=(1, 1))
    print("ConvModule (1,1)x64 Average Time: {:.3f}".format(model_perf(model, self.x)))
    print("ConvModule (1,1)x64 Average Train Time: {:.3f}".format(model_train_perf(model, self.x, self.y)))
    print("ConvModule (1,1)x64 Parameters: {:,d}".format(model_params(model)))

    model = ConvModule(128, kernel_size=(3, 3))
    print("ConvModule (3,3)x128 Average Time: {:.3f}".format(model_perf(model, self.x)))
    print("ConvModule (3,3)128 Average Train Time: {:.3f}".format(model_train_perf(model, self.x, self.y)))
    print("ConvModule (3,3)x128 Parameters: {:,d}".format(model_params(model)))

    model = ConvModule(128, kernel_size=(1, 1))
    print("ConvModule (1,1)x128 Average Time: {:.3f}".format(model_perf(model, self.x)))
    print("ConvModule (1,1)128 Average Train Time: {:.3f}".format(model_train_perf(model, self.x, self.y)))
    print("ConvModule (1,1)x128 Parameters: {:,d}".format(model_params(model)))

  # ConvModule (3,3)x64 Average Time: 0.090
  # ConvModule (3,3)x64 Parameters: 7,682,048
  # ConvModule (1,1)x64 Average Time: 0.084
  # ConvModule (1,1)x64 Parameters: 7,680,512
  # ConvModule (3,3)x128 Average Time: 0.169
  # ConvModule (3,3)x128 Parameters: 15,364,096
  # ConvModule (1,1)x128 Average Time: 0.165
  # ConvModule (1,1)x128 Parameters: 15,361,024

  # @unittest.skip("")
  def test_SqueezeNetFire_module(self):
    model = SqueezeNetFireModule(16, 64)
    print("SqueezeNetFireModule 16x64 Average Time: {:.3f}".format(model_perf(model, self.x)))
    # print("SqueezeNetFireModule 16x64 Parameters: {:,d}".format(model_params(model)))
    #
    # model = SqueezeNetFireModule(32, 128)
    # print("SqueezeNetFireModule 32x128 Average Time: {:.3f}".format(model_perf(model, self.x)))
    # print("SqueezeNetFireModule 32x128 Parameters: {:,d}".format(model_params(model)))

    # print("saving")
    # model.save("testing_sqeeze_save.h5")
    # print("loading")
    # model.save("testing_sqeeze_save.h5")
# SqueezeNetFireModule 16x64 Average Time: 0.062
# SqueezeNetFireModule 16x64 Parameters: 10,816
# SqueezeNetFireModule 32x128 Average Time: 0.136
# SqueezeNetFireModule 32x128 Parameters: 42,112

if __name__ == '__main__':
    unittest.main()