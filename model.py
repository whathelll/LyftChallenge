import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time

class InitialModule(tf.keras.Model):
  def __init__(self, filters):
    super(InitialModule, self).__init__()
    self.conv = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same', strides=(2, 2))
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
  def __init__(self, reduce_depth, expand_depth, dropout_rate=0., down_sample=False, factorize=False):
    super(ConvModule, self).__init__()
    self.down_sample = down_sample
    self.factorize = factorize
    strides = (1, 1)
    if down_sample:
      strides=(2, 2)
      self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=strides, padding='same')

    self.conv1 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(1, 1), padding='same', use_bias=False)
    self.prelu1 = tf.keras.layers.PReLU()
    self.bn1 = tf.keras.layers.BatchNormalization()

    if factorize:
      self.conv21 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(5, 1), strides=strides, padding='same')
      self.conv22 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(1, 5), strides=strides, padding='same')
    else:
      self.conv2 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(3, 3), strides=strides, padding='same')

    self.prelu2 = tf.keras.layers.PReLU()
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(1, 1), padding='same')
    self.prelu3 = tf.keras.layers.PReLU()
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.prelu_last = tf.keras.layers.PReLU()

    self.concat = tf.keras.layers.Concatenate()

  def call(self, inputs, training=False):
    if self.down_sample:
      main = self.maxpool(inputs)
    else:
      main = inputs

    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.prelu1(x)

    if self.factorize:
      x = self.conv21(x)
      x = self.conv22(x)
    else:
      x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.prelu2(x)

    x = self.conv3(x)
    x = self.bn3(x, training=training)
    x = self.prelu3(x)

    x = self.regularizer(x, training=training)

    if self.down_sample:
      x = self.concat([x, main])
    else:
      x = tf.add(x, main)
      x = self.prelu_last(x)
    return x

class DilatedModule(tf.keras.Model):
  def __init__(self, reduce_depth, expand_depth, dropout_rate, dilation_rate=(2, 2)):
    super(DilatedModule, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(1, 1), padding='same', use_bias=False)
    self.prelu1 = tf.keras.layers.PReLU()
    self.bn1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(3, 3), padding='same', dilation_rate=dilation_rate)
    self.prelu2 = tf.keras.layers.PReLU()
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(1, 1), padding='same')
    self.prelu3 = tf.keras.layers.PReLU()
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.prelu_last = tf.keras.layers.PReLU()


  def call(self, inputs, training=False):
    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.prelu1(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.prelu2(x)

    x = self.conv3(x)
    x = self.bn3(x, training=training)
    x = self.prelu3(x)

    x = self.regularizer(x, training=training)

    x = tf.add(x, inputs)
    x = self.prelu_last(x)
    return x

class UpSampleModule(tf.keras.Model):
  def __init__(self, reduce_depth, expand_depth, dropout_rate):
    super(UpSampleModule, self).__init__()
    # self.upconv1 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(1, 1), padding='same')
    self.upbn1 = tf.keras.layers.BatchNormalization()
    self.upsample = tf.keras.layers.Conv2DTranspose(expand_depth, kernel_size=(3, 3), strides=(2, 2), padding='same')

    self.conv1 = tf.keras.layers.Conv2D(reduce_depth, kernel_size=(1, 1), padding='same', use_bias=False)
    self.prelu1 = tf.keras.layers.PReLU()
    self.bn1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2DTranspose(reduce_depth, kernel_size=(3, 3), strides=(2, 2), padding='same')
    self.prelu2 = tf.keras.layers.PReLU()
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(expand_depth, kernel_size=(1, 1), padding='same')
    self.prelu3 = tf.keras.layers.PReLU()
    self.bn3 = tf.keras.layers.BatchNormalization()

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.prelu_last = tf.keras.layers.PReLU()


  def call(self, inputs, training=False):
    # up = self.upconv1(inputs)
    up = self.upsample(inputs)
    up = self.upbn1(up, training=training)

    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.prelu1(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.prelu2(x)

    x = self.conv3(x)
    x = self.bn3(x, training=training)
    x = self.prelu3(x)

    x = self.regularizer(x, training=training)

    x = tf.add(x, up)
    x = self.prelu_last(x)
    return x



class DilatedCNN(tf.keras.Model):
  def __init__(self):
    super(DilatedCNN, self).__init__()
    self.car_weight = 8.0
    self.initial = InitialModule(filters=13)

    self.down1_1 = ConvModule(reduce_depth=4, expand_depth=16, dropout_rate=0.01, down_sample=True)
    self.down1_2 = ConvModule(reduce_depth=8, expand_depth=32, dropout_rate=0.01, down_sample=False)
    self.down1_3 = ConvModule(reduce_depth=8, expand_depth=32, dropout_rate=0.01, down_sample=False)
    self.down1_4 = ConvModule(reduce_depth=8, expand_depth=32, dropout_rate=0.01, down_sample=False)
    self.down1_5 = ConvModule(reduce_depth=8, expand_depth=32, dropout_rate=0.01, down_sample=False)

    self.down2_0 = ConvModule(reduce_depth=16, expand_depth=32, dropout_rate=0.1, down_sample=True)
    self.down2_1 = ConvModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, down_sample=False)
    self.down2_2 = DilatedModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, dilation_rate=(2, 2))
    self.down2_3 = ConvModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, down_sample=False, factorize=True)
    self.down2_4 = DilatedModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, dilation_rate=(4, 4))
    self.down2_5 = ConvModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, down_sample=False)
    self.down2_6 = DilatedModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, dilation_rate=(8, 8))
    self.down2_7 = ConvModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, down_sample=False, factorize=True)
    self.down2_8 = DilatedModule(reduce_depth=16, expand_depth=64, dropout_rate=0.1, dilation_rate=(16, 16))

    # should we repeat section 2 without down2_0?

    self.up1_concat = tf.keras.layers.Concatenate()
    self.up1_0 = UpSampleModule(reduce_depth=16, expand_depth=32, dropout_rate=0.1)
    self.up1_1 = ConvModule(reduce_depth=8, expand_depth=32, dropout_rate=0.1, down_sample=False)
    self.up1_2 = ConvModule(reduce_depth=8, expand_depth=32, dropout_rate=0.1, down_sample=False)

    self.up2_concat = tf.keras.layers.Concatenate()
    self.up2_0 = UpSampleModule(reduce_depth=8, expand_depth=16, dropout_rate=0.1)
    self.up2_1 = ConvModule(reduce_depth=4, expand_depth=16, dropout_rate=0.1, down_sample=False)

    self.up3_concat = tf.keras.layers.Concatenate()
    self.up3_0 = UpSampleModule(reduce_depth=8, expand_depth=16, dropout_rate=0.1)
    self.final = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), padding='same', activation=tf.nn.softmax)
    self.final_concat = tf.keras.layers.Concatenate()
    self.final_up = tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.softmax)

  def call(self, inputs, training=False):
    # now = time.time()
    init = self.initial(inputs, training=training)
    # print("init:", init.shape)
    # print(time.time() - now)

    down1 = self.down1_1(init, training=training)
    down1 = self.down1_2(down1, training=training)
    down1 = self.down1_3(down1, training=training)
    down1 = self.down1_4(down1, training=training)
    down1 = self.down1_5(down1, training=training)
    # print("down1:", down1.shape)
    # print(time.time() - now)

    down2 = self.down2_0(down1, training=training)
    down2 = self.down2_1(down2, training=training)
    down2 = self.down2_2(down2, training=training)
    down2 = self.down2_3(down2, training=training)
    down2 = self.down2_4(down2, training=training)
    down2 = self.down2_5(down2, training=training)
    down2 = self.down2_6(down2, training=training)
    down2 = self.down2_7(down2, training=training)
    down2 = self.down2_8(down2, training=training)
    # print("down2:", down2.shape)
    # print(time.time() - now)

    up1 = self.up1_0(down2, training=training)
    up1 = self.up1_1(up1, training=training)
    up1 = self.up1_2(up1, training=training)
    up1 = self.up1_concat([up1, down1])
    # print("up1:", up1.shape)
    # print(time.time() - now)

    up2 = self.up2_0(up1, training=training)
    up2 = self.up2_1(up2, training=training)
    up2 = self.up2_concat([up2, init])
    # print("up2:", up2.shape)
    # print(time.time() - now)

    up3 = self.up3_0(up2, training=training)
    up3 = self.final_concat([up3, inputs])
    x = self.final(up3)
    # x = self.final_up(up2)

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

    loss = (self.car_weight*car + road + nothing)
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

  def save(self, name="model_weights.h5"):
    self.save_weights(name)

  def load(self, name="model_weights.h5"):
    x = tf.random_normal((1, 320, 400, 3))
    self(x)
    self.load_weights(name)

