import tensorflow as tf
import tensorflow.contrib.eager as tfe


class DownSample(tf.keras.Model):
  def __init__(self, filters):
    super(DownSample, self).__init__()
    self.conv = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same', strides=(2, 2))
    self.prelu = tf.keras.layers.PReLU()
    self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
    self.concat = tf.keras.layers.Concatenate()

  def call(self, inputs):
    pool = self.maxpool(inputs)
    x = self.conv(inputs)
    x = self.prelu(x)
    x = self.concat([x, pool])
    return x

class DilationLayer(tf.keras.Model):
  def __init__(self, filters):
    super(DilationLayer, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', dilation_rate=(1, 1))
    self.prelu1 = tf.keras.layers.PReLU()
    self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same', dilation_rate=(2, 2))
    self.prelu2 = tf.keras.layers.PReLU()
    self.batchnorm = tf.keras.layers.BatchNormalization()

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.prelu1(x)
    x = self.conv2(x)
    x = self.prelu2(x)
    x = self.batchnorm(x)
    return x

class UpSample(tf.keras.Model):
  def __init__(self, filters):
    super(UpSample, self).__init__()
    self.conv1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')
    self.prelu1 = tf.keras.layers.PReLU()
    self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same', dilation_rate=(1, 1))
    self.prelu2 = tf.keras.layers.PReLU()

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.prelu1(x)
    x = self.conv2(x)
    x = self.prelu2(x)
    return x




class DilatedCNN(tf.keras.Model):
  def __init__(self):
    super(DilatedCNN, self).__init__()
    self.down1 = DownSample(filters=13)
    self.down2 = DownSample(filters=24)
    self.down3 = DownSample(filters=36)

    self.dilation1 = DilationLayer(filters=76)
    self.dilation2 = DilationLayer(filters=76)
    self.dilation3 = DilationLayer(filters=76)

    self.up1 = UpSample(filters=36)
    self.up1_concat = tf.keras.layers.Concatenate()

    self.up2 = UpSample(filters=18)
    self.up2_concat = tf.keras.layers.Concatenate()

    self.last_concat = tf.keras.layers.Concatenate()
    self.last_conv1 = tf.keras.layers.Conv2DTranspose(9, kernel_size=(3, 3), strides=(2, 2), padding='same')
    self.last_prelu = tf.keras.layers.PReLU()
    self.last_layer = tf.keras.layers.Conv2D(2, kernel_size=(3, 3),
                                             padding='same', dilation_rate=(1, 1), activation=tf.nn.sigmoid)
    #         self.up3 = UpSample(filters=2)

    self.beta_road = tf.constant(0.5)
    self.beta_car = tf.constant(2.)

  def call(self, inputs):
    d1 = self.down1(inputs)
    d2 = self.down2(d1)
    x = self.down3(d2)
    x = self.dilation1(x)
    x = self.dilation2(x)
    x = self.dilation3(x)
    x = self.up1(x)
    x = self.up1_concat([x, d2])
    x = self.up2(x)
    x = self.up2_concat([x, d1])
    x = self.last_conv1(x)
    x = self.last_concat([x, inputs])
    x = self.last_layer(x)
    return x

  def loss(self, predictions, targets):
    """road = layer [0], car = layer [1]"""
    y = self(inputs)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=y)
    return loss

  def loss_dice_coef(self, y_hat, y):
    #         y = tf.reshape(y, [-1])
    #         y_hat = tf.reshape(y_hat, [-1])

    intersection = tf.reduce_sum(y * y_hat)
    top = 2 * intersection + 1
    bottom = tf.reduce_sum(y) + tf.reduce_sum(y_hat) + 1
    return -top / bottom
  #         return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_hat)

  def grad(self, x, y):
    with tfe.GradientTape() as tape:
      y_hat = self(x)
      loss_value = self.loss_dice_coef(y_hat, y)
    return tape.gradient(loss_value, self.variables)

  def train(self, x, y, optimizer):
    grads = self.grad(x, y)
    optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())

  def save(self):
    self.save_weights("model_weights.h5")

  def load(self):
    self.load_weights("model_weights.h5")


