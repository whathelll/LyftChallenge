import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time

class DownModule(tf.keras.Model):
  def __init__(self, filters):
    super(DownModule, self).__init__()
    self.conv = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=tf.nn.relu)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
    self.concat = tf.keras.layers.Concatenate()

  def call(self, inputs, training=False):
    pool = self.maxpool(inputs)
    x = self.conv(inputs)
    x = self.bn1(x, training=training)
    x = self.concat([x, pool])
    return x

class ConvModule(tf.keras.Model):
  def __init__(self, reduce_depth, expand_depth, dropout_rate=0.):
    super(ConvModule, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(1, 1), padding='same', use_bias=False, activation=tf.nn.relu)
    self.bn1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.concat = tf.keras.layers.Concatenate()

  def call(self, inputs, training=False):
    x = inputs
    x = self.conv1(x)
    x = self.bn1(x, training=training)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    x = self.conv3(x)
    x = self.bn3(x, training=training)

    x = self.regularizer(x, training=training)

    x = tf.add(x, inputs)
    return x

class DilatedModule(tf.keras.Model):
  def __init__(self, reduce_depth, expand_depth, dropout_rate, dilation_rate=(2, 2)):
    super(DilatedModule, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(1, 1), padding='same', use_bias=False, activation=tf.nn.relu)
    self.bn1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(3, 3), padding='same', dilation_rate=dilation_rate)
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(1, 1), padding='same')
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)


  def call(self, inputs, training=False):
    x = inputs
    x = self.conv1(x)
    x = self.bn1(x, training=training)

    x = self.conv2(x)
    x = self.bn2(x, training=training)

    x = self.conv3(x)
    x = self.bn3(x, training=training)

    x = self.regularizer(x, training=training)

    x = tf.add(x, inputs)
    return x

class UpSampleModule(tf.keras.Model):
  def __init__(self, expand_depth, dropout_rate):
    super(UpSampleModule, self).__init__()
    self.upsample = tf.keras.layers.Conv2DTranspose(expand_depth, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
    self.upbn1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)


  def call(self, inputs, training=False):
    # up = self.upconv1(inputs)
    up = self.upsample(inputs)
    up = self.upbn1(up, training=training)

    x = up
    x = self.conv2(x)
    x = self.bn2(x, training=training)

    x = self.conv3(x)
    x = self.bn3(x, training=training)

    x = self.regularizer(x, training=training)
    x = tf.add(x, up)
    return x


class DilatedCNN(tf.keras.Model):
  def __init__(self):
    super(DilatedCNN, self).__init__()
    self.car_weight = 4.0
    self.down1 = DownModule(filters=13)

    self.conv1_0 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
    self.conv1_1 = ConvModule(16, 32, dropout_rate=0.05)
    self.conv1_2 = ConvModule(16, 32, dropout_rate=0.05)
    self.conv1_3 = ConvModule(16, 32, dropout_rate=0.05)

    self.down2 = DownModule(filters=64)
    self.conv2_0 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
    self.conv2_1 = ConvModule(16, 64, dropout_rate=0.1)
    self.conv2_2 = ConvModule(16, 64, dropout_rate=0.1)
    self.conv2_3 = ConvModule(16, 64, dropout_rate=0.1)

    self.down3 = DownModule(filters=128)
    self.conv3_0 = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
    self.conv3_1 = ConvModule(32, 128, dropout_rate=0.1)
    self.conv3_2 = DilatedModule(32, 128, dropout_rate=0.1, dilation_rate=(2, 2))
    self.conv3_3 = DilatedModule(32, 128, dropout_rate=0.1, dilation_rate=(4, 4))
    self.conv3_4 = DilatedModule(32, 128, dropout_rate=0.1, dilation_rate=(8, 8))
    self.conv3_5 = DilatedModule(32, 128, dropout_rate=0.1, dilation_rate=(16, 16))


    self.up2 = UpSampleModule(64, dropout_rate=0.)
    self.concat_up2 = tf.keras.layers.Concatenate()
    self.conv_up2_0 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
    self.conv_up2_1 = ConvModule(16, 64, dropout_rate=0.)
    self.conv_up2_2 = ConvModule(16, 64, dropout_rate=0.)

    self.up1 = UpSampleModule(32, dropout_rate=0.)
    self.concat_up1 = tf.keras.layers.Concatenate()
    self.conv_up1_0 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', activation=tf.nn.relu)
    self.conv_up1_1 = ConvModule(8, 32, dropout_rate=0.)
    self.conv_up1_2 = ConvModule(8, 32, dropout_rate=0.)

    self.up0 = UpSampleModule(16, dropout_rate=0.)
    self.concat_up0 = tf.keras.layers.Concatenate()
    self.final_0 = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    self.final_1 = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), padding='same', activation=tf.nn.softmax)


  def call(self, inputs, training=False):
    # now = time.time()
    d1 = self.down1(inputs, training=training)
    d1_x = self.conv1_0(d1)
    d1_x = self.conv1_1(d1_x)
    d1_x = self.conv1_2(d1_x)
    # d1_x = self.conv1_3(d1_x)

    d2 = self.down2(d1_x)
    d2_x = self.conv2_0(d2)
    d2_x = self.conv2_1(d2_x)
    d2_x = self.conv2_2(d2_x)
    # d2_x = self.conv2_3(d2_x)

    d3 = self.down3(d2_x)
    d3_x = self.conv3_0(d3)
    d3_x = self.conv3_1(d3_x)
    d3_x = self.conv3_2(d3_x)
    d3_x = self.conv3_3(d3_x)
    d3_x = self.conv3_4(d3_x)
    d3_x = self.conv3_5(d3_x)

    up2 = self.up2(d3_x)
    up2 = self.concat_up2([up2, d2_x])
    up2 = self.conv_up2_0(up2)
    up2 = self.conv_up2_1(up2)
    # up2 = self.conv_up2_2(up2)

    up1 = self.up1(up2)
    up1 = self.concat_up2([up1, d1_x])
    up1 = self.conv_up1_0(up1)
    # up1 = self.conv_up1_1(up1)
    # up1 = self.conv_up1_2(up1)

    up0 = self.up0(up1)
    # up0 = self.concat_up0([up0, inputs])
    up0 = self.final_0(up0)
    up0 = self.final_1(up0)

    x = up0
    # print(time.time() - now)
    return x

  def loss(self, y_hat, y):
    """road = layer [0], car = layer [1]"""
    road_y_hat = y_hat[:, :, :, 0]
    car_y_hat = y_hat[:, :, :, 1]
    nothing_y_hat = y_hat[:, :, :, 2]
    road_y = y[:, :, :, 0]
    car_y = y[:, :, :, 1]
    nothing_y = y[:, :, :, 2]
    car = self.loss_dice_coef(car_y_hat, car_y)
    road = self.loss_dice_coef(road_y_hat, road_y)
    nothing = self.loss_dice_coef(nothing_y_hat, nothing_y)

    loss = (car + road + nothing)
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
      y_hat = self(x, training=True)
      loss_value = self.loss(y_hat, y)
      mean_loss = tf.reduce_mean(loss_value)
    return tape.gradient(mean_loss, self.variables), loss_value

  def train(self, x, y, optimizer):
    grads, loss_value = self.grad(x, y)
    optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
    return loss_value

  def save(self, name="model_v2_weights.h5"):
    self.save_weights(name)

  def load(self, name="model_v2_weights.h5"):
    x = tf.random_normal((1, 320, 400, 3))
    self(x)
    self.load_weights(name)

