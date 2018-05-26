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

  def loss(self, y_hat, y):
    """road = layer [0], car = layer [1]"""
    road_y_hat = y_hat[:, :, :, 0]
    car_y_hat = y_hat[:, :, :, 1]
    road_y = y[:, :, :, 0]
    car_y = y[:, :, :, 1]
    car = self.loss_dice_coef(car_y_hat, car_y)
    road = self.loss_dice_coef(road_y_hat, road_y)

    loss = (9.*car + road)/10.
    return loss


  def loss_dice_coef(self, y_hat, y, beta=1.):
    # beta = beta ** 2
    y = tf.reshape(y, [y.shape[0], -1])
    y_hat = tf.reshape(y_hat, [y_hat.shape[0], -1])

    intersection = tf.reduce_sum(y * y_hat, axis=1, keep_dims=True)
    top = 2. * intersection + 1.

    # top = (1 + beta) * top
    # print("intersection", intersection.shape)
    bottom = tf.reduce_sum(y, axis=1, keep_dims=True) + tf.reduce_sum(y_hat, axis=1, keep_dims=True) + 1.

    # bottom = beta * bottom + 1
    return 1. - top / bottom

  def grad(self, x, y):
    with tfe.GradientTape() as tape:
      y_hat = self(x)
      loss_value = self.loss(y_hat, y)
      mean_loss = tf.reduce_mean(loss_value)
    return tape.gradient(mean_loss, self.variables), loss_value

  def train(self, x, y, optimizer):
    grads, loss_value = self.grad(x, y)
    optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
    return loss_value

  def save(self, name="model_weights.h5"):
    self.save_weights(name)

  def load(self, name="model_weights.h5"):
    x = tf.random_normal((1, 600, 800, 3))
    self(x)
    self.load_weights(name)

