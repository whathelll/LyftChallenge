import tensorflow as tf

"""Standard Convolution module"""
class ConvModule(tf.keras.Model):
  def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1),
               use_bias=True, *args, **kwargs):
    super().__init__(self, *args, **kwargs)
    self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                        dilation_rate=dilation_rate, use_bias=use_bias)
    self.bn1 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training=False):
    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    return x

"""Convolution module which does 3x1 and then 1x3 etc"""
class SpatialConvModule(tf.keras.Model):
  def __init__(self, filters, kernel_size=3, strides=(1, 1), padding='same', dilation_rate=(1, 1),
               use_bias=True, *args, **kwargs):
    super().__init__(self, *args, **kwargs)
    self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(kernel_size, 1), strides=strides, padding=padding,
                                        dilation_rate=dilation_rate, use_bias=use_bias)
    self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(1, kernel_size), strides=strides, padding=padding,
                                        dilation_rate=dilation_rate, use_bias=use_bias)
    self.bn1 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training=False):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    return x

"""Squeeze Net Fire module"""
class SqueezeNetFireModule(tf.keras.Model):
  def __init__(self, reduce_filters, expand_filters, *args, **kwargs):
    super().__init__(self, *args, **kwargs)
    self.squeeze = tf.keras.layers.Conv2D(reduce_filters, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    self.left = tf.keras.layers.Conv2D(expand_filters, kernel_size=(1, 1), padding="same", activation=tf.nn.relu)
    self.right = tf.keras.layers.Conv2D(expand_filters, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    self.concat = tf.keras.layers.Concatenate()

  def call(self, inputs, training=False):
    x = self.squeeze(inputs)
    left = self.left(x)
    right = self.right(x)
    return self.concat([left, right])

"""Module which does a number of convolutions and then do max pooling"""
class ConvPoolModule(tf.keras.Model):
  def __init__(self, filters, conv_layers=2, *args, **kwargs):
    super().__init__(self, *args, **kwargs)
    self.conv_layers = conv_layers
    self.convs = []
    for i in range(self.conv_layers):
      self.convs.append(ConvModule(filters))
    self.max_pool = tf.keras.layers.MaxPool2D()

  def call(self, inputs, training=False):
    x = inputs
    for i in range(self.conv_layers):
      x = self.convs[i](x)
    before_pool = x
    x = self.max_pool(x)
    return x, before_pool

