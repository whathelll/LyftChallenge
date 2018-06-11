import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
from modules import ConvModule
from loss import Loss

class InitialModule(tf.keras.Model):
  def __init__(self, filters):
    super(InitialModule, self).__init__()
    self.conv = ConvModule(filters=filters, kernel_size=(3, 3), strides=(2, 2))
    self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
    self.concat = tf.keras.layers.Concatenate()

  def call(self, inputs, training=False):
    pool = self.maxpool(inputs)
    x = self.conv(inputs, training=training)
    x = self.concat([x, pool])
    return x

class ConvolutionModule(tf.keras.Model):
  def __init__(self, reduce_depth, expand_depth, dropout_rate=0., down_sample=False, factorize=False):
    super(ConvolutionModule, self).__init__()
    self.down_sample = down_sample
    self.factorize = factorize
    strides = (1, 1)
    if down_sample:
      strides=(2, 2)
      self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=strides, padding='same')

    self.conv1 = ConvModule(reduce_depth, kernel_size=(1, 1))

    if factorize:
      self.conv21 = ConvModule(expand_depth, kernel_size=(5, 1), strides=strides)
      self.conv22 = ConvModule(expand_depth, kernel_size=(1, 5), strides=strides)
    else:
      self.conv2 = ConvModule(expand_depth, kernel_size=(3, 3), strides=strides)

    self.conv3 = ConvModule(expand_depth, kernel_size=(1, 1))

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.prelu_last = tf.keras.layers.PReLU()

    self.concat = tf.keras.layers.Concatenate()
    self.conv4 = ConvModule(expand_depth, kernel_size=(1, 1))

  def call(self, inputs, training=False):
    if self.down_sample:
      main = self.maxpool(inputs)
    else:
      main = inputs
    x = inputs
    # x = self.conv1(x, training=training)

    if self.factorize:
      x = self.conv21(x, training=training)
      x = self.conv22(x, training=training)
    else:
      x = self.conv2(x, training=training)

    # x = self.conv3(x, training=training)

    x = self.regularizer(x, training=training)

    if self.down_sample:
      x = self.concat([x, main])
      x = self.conv4(x, training=training)
    else:
      x = tf.add(x, main)
      x = self.prelu_last(x)
    return x

class DilatedModule(tf.keras.Model):
  def __init__(self, reduce_depth, expand_depth, dropout_rate, dilation_rate=(2, 2)):
    super(DilatedModule, self).__init__()
    self.conv1 = ConvModule(reduce_depth, kernel_size=(1, 1))
    self.conv2 = ConvModule(expand_depth, kernel_size=(3, 3), dilation_rate=dilation_rate)
    self.conv3 = ConvModule(expand_depth, kernel_size=(1, 1))

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.prelu_last = tf.keras.layers.PReLU()


  def call(self, inputs, training=False):
    x = inputs
    # x = self.conv1(x, training=training)
    x = self.conv2(x, training=training)
    # x = self.conv3(x, training=training)

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
    self.prelu0 = tf.keras.layers.PReLU()

    self.conv1 = ConvModule(reduce_depth, kernel_size=(1, 1))

    self.conv2 = tf.keras.layers.Conv2DTranspose(expand_depth, kernel_size=(3, 3), strides=(2, 2), padding='same')
    self.prelu2 = tf.keras.layers.PReLU()
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv3 = ConvModule(expand_depth, kernel_size=(1, 1))

    self.regularizer = tf.keras.layers.SpatialDropout2D(dropout_rate)
    self.prelu_last = tf.keras.layers.PReLU()


  def call(self, inputs, training=False):
    # up = self.upconv1(inputs)
    up = self.upsample(inputs)
    up = self.upbn1(up, training=training)
    up = self.prelu0(up)

    x = inputs
    # x = self.conv1(inputs, training=training)

    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.prelu2(x)

    # x = self.conv3(x, training=training)

    x = self.regularizer(x, training=training)

    x = tf.add(x, up)
    x = self.prelu_last(x)
    return x



class DilatedCNN(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dice_loss = True
    self.initial = InitialModule(filters=29)

    self.down1_1 = ConvolutionModule(reduce_depth=64, expand_depth=64, dropout_rate=0.01, down_sample=True)
    self.down1_2 = ConvolutionModule(reduce_depth=64, expand_depth=64, dropout_rate=0.01, down_sample=False)
    self.down1_3 = ConvolutionModule(reduce_depth=64, expand_depth=64, dropout_rate=0.01, down_sample=False)
    self.down1_4 = ConvolutionModule(reduce_depth=64, expand_depth=64, dropout_rate=0.01, down_sample=False)
    self.down1_5 = ConvolutionModule(reduce_depth=64, expand_depth=64, dropout_rate=0.01, down_sample=False)

    self.down2_0 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=True)
    self.down2_1 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False)
    self.down2_2 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(2, 2))
    self.down2_3 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False, factorize=True)
    self.down2_4 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(4, 4))
    self.down2_5 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False)
    self.down2_6 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(8, 8))
    self.down2_7 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False, factorize=True)
    self.down2_8 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(16, 16))
    self.down2_9 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(8, 8))
    self.down2_10 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(4, 4))
    self.down2_11 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(2, 2))
    self.down2_12 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(1, 1))

    # self.down3_0 = ConvolutionModule(reduce_depth=32, expand_depth=128, dropout_rate=0.1, down_sample=True)
    self.down3_1 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False)
    self.down3_2 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(2, 2))
    self.down3_3 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False, factorize=True)
    self.down3_4 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(4, 4))
    self.down3_5 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False)
    self.down3_6 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(8, 8))
    self.down3_7 = ConvolutionModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, down_sample=False, factorize=True)
    self.down3_8 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(16, 16))
    self.down3_9 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(8, 8))
    self.down3_10 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(4, 4))
    self.down3_11 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(2, 2))
    self.down3_12 = DilatedModule(reduce_depth=128, expand_depth=128, dropout_rate=0.1, dilation_rate=(1, 1))

    # should we repeat section 2 without down2_0?

    self.up1_concat = tf.keras.layers.Concatenate()
    self.up1_0_1x1 = ConvModule(64, kernel_size=(1, 1))
    self.up1_0 = UpSampleModule(reduce_depth=64, expand_depth=64, dropout_rate=0.1)
    self.up1_1 = ConvolutionModule(reduce_depth=64, expand_depth=64, dropout_rate=0.1, down_sample=False)
    self.up1_2 = ConvolutionModule(reduce_depth=64, expand_depth=64, dropout_rate=0.1, down_sample=False)

    self.up2_concat = tf.keras.layers.Concatenate()
    self.up2_0_1x1 = ConvModule(32, kernel_size=(1, 1))
    self.up2_0 = UpSampleModule(reduce_depth=32, expand_depth=32, dropout_rate=0.1)
    self.up2_1 = ConvolutionModule(reduce_depth=32, expand_depth=32, dropout_rate=0.1, down_sample=False)

    self.up3_0 = UpSampleModule(reduce_depth=32, expand_depth=32, dropout_rate=0.1)
    self.final_0 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    self.final_1 = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), padding='same', activation=tf.nn.softmax)
    self.final_concat = tf.keras.layers.Concatenate()
    # self.final_up = tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=tf.nn.softmax)

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
    # down2 = self.down2_8(down2, training=training)
    # down2 = self.down2_9(down2, training=training)
    # down2 = self.down2_10(down2, training=training)
    # down2 = self.down2_11(down2, training=training)
    # down2 = self.down2_12(down2, training=training)
    # print("down2:", down2.shape)
    # print(time.time() - now)

    down2 = self.down3_1(down2, training=training)
    down2 = self.down3_2(down2, training=training)
    down2 = self.down2_3(down2, training=training)
    down2 = self.down3_4(down2, training=training)
    down2 = self.down2_5(down2, training=training)
    down2 = self.down3_6(down2, training=training)
    down2 = self.down2_7(down2, training=training)
    # down2 = self.down3_8(down2, training=training)
    # down2 = self.down2_9(down2, training=training)
    # down2 = self.down2_10(down2, training=training)
    # down2 = self.down2_11(down2, training=training)
    # down2 = self.down2_12(down2, training=training)


    up1 = self.up1_0(down2, training=training)
    up1 = self.up1_concat([up1, down1])
    up1 = self.up1_0_1x1(up1)
    up1 = self.up1_1(up1, training=training)
    # up1 = self.up1_2(up1, training=training)
    # print("up1:", up1.shape)
    # print(time.time() - now)

    up2 = self.up2_0(up1, training=training)
    up2 = self.up2_concat([up2, init])
    up2 = self.up2_0_1x1(up2)
    up2 = self.up2_1(up2, training=training)
    # print("up2:", up2.shape)
    # print(time.time() - now)

    up3 = self.up3_0(up2, training=training)
    up3 = self.final_concat([up3, inputs])
    up3 = self.final_0(up3)
    x = self.final_1(up3)
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

    if self.dice_loss:
      loss = (1.3*car + road + 0.7*nothing)
    else:
      loss = (y - y_hat) ** 2
    return loss, car, road


  def loss_dice_coef(self, y_hat, y):
    # beta = beta ** 2
    y = tf.reshape(y, [tf.shape(y)[0], -1])
    y_hat = tf.reshape(y_hat, [tf.shape(y_hat)[0], -1])

    intersection = tf.reduce_sum(y * y_hat, axis=1, keepdims=True)
    top = 2. * intersection + 1.
    bottom = tf.reduce_sum(y, axis=1, keepdims=True) + tf.reduce_sum(y_hat, axis=1, keepdims=True) + 1.
    return 1. - top / bottom

  def grad(self, x, y):
    with tfe.GradientTape() as tape:
      y_hat = self(x, training=True)
      loss_value, car_loss, road_loss = self.loss(y_hat, y)
      mean_loss = tf.reduce_mean(loss_value)
    return tape.gradient(mean_loss, self.variables), loss_value,  car_loss, road_loss

  def train(self, x, y, optimizer):
    grads, loss_value, car_loss, road_loss = self.grad(x, y)
    optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
    return loss_value, car_loss, road_loss

  @staticmethod
  def build_keras_model():
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    x = tf.keras.layers.Input(shape=(320, 400, 3))
    y_hat = DilatedCNN()(x)

    _keras_model = tf.keras.Model(inputs=x, outputs=y_hat)
    _keras_model.compile(optimizer=optimizer, loss=Loss.Lyft_loss,
                              metrics=[Loss.Lyft_FScore, Loss.Lyft_car_Fscore, Loss.Lyft_road_Fscore])
    return _keras_model


  def save(self, name="model_v4_weights.h5"):
    print("saving")
    self.save_weights(name)

  def load(self, name="model_v4_weights.h5"):
    x = tf.random_normal((1, 320, 400, 3))
    self(x)
    self.load_weights(name)

def keras_loss_dice_coef(y, y_hat):
  y = tf.keras.backend.flatten(y)
  y_hat = tf.keras.backend.flatten(y_hat)
  intersection = tf.keras.backend.sum(y * y_hat)
  top = (2. * intersection + 1)
  bottom = (tf.keras.backend.sum(y) + tf.keras.backend.sum(y_hat) + 1)
  return 1. - top / bottom

