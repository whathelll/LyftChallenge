import tensorflow as tf
from modules import ConvModule, SpatialConvModule
from loss import Loss


class UNetConvAndDownModule(tf.keras.Model):
  def __init__(self, filters=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.conv0 = SpatialConvModule(filters)
    # self.conv1 = SpatialConvModule(filters)
    self.conv0 = ConvModule(filters)
    self.conv1 = ConvModule(filters)

  def call(self, inputs, training=False):
    d1 = self.conv0(inputs, training=training)
    d1 = self.conv1(d1, training=training)
    return d1

class UNetConvAndUpModule(tf.keras.Model):
  def __init__(self, filters=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv0 = ConvModule(filters)
    self.conv1 = ConvModule(filters)
    self.upsample = tf.keras.layers.UpSampling2D()

  def call(self, inputs, training=False):
    u1 = self.conv0(inputs, training=training)
    u1 = self.conv1(u1, training=training)
    u1 = self.upsample(u1)
    return u1


class UNet(tf.keras.Model):
  def __init__(self, start_filter=64, *args, **kwargs):
    super().__init__(*args, **kwargs)
    filters = start_filter
    self.down1 = UNetConvAndDownModule(filters=filters)
    self.pool1 = tf.keras.layers.MaxPool2D()
    self.down2 = UNetConvAndDownModule(filters=filters * 2)
    self.pool2 = tf.keras.layers.MaxPool2D()
    self.down3 = UNetConvAndDownModule(filters=filters * 4)
    self.pool3 = tf.keras.layers.MaxPool2D()
    self.down4 = UNetConvAndDownModule(filters=filters * 8)
    self.pool4 = tf.keras.layers.MaxPool2D()

    self.up5 = UNetConvAndUpModule(filters=filters*16)

    self.concat4 = tf.keras.layers.Concatenate()
    self.up4 = UNetConvAndUpModule(filters=filters*8)
    self.concat3 = tf.keras.layers.Concatenate()
    self.up3 = UNetConvAndUpModule(filters=filters*4)
    self.concat2 = tf.keras.layers.Concatenate()
    self.up2 = UNetConvAndUpModule(filters=filters*2)
    self.concat1 = tf.keras.layers.Concatenate()

    self.up1_conv0 = ConvModule(filters=filters)
    self.up1_conv1 = ConvModule(filters=filters)
    self.up1_conv2 = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), padding='same', activation=tf.nn.softmax)

  def call(self, inputs, training=False):
    d1= self.down1(inputs, training=training)
    pool1 = self.pool1(d1)
    d2= self.down2(pool1, training=training)
    pool2 = self.pool2(d2)
    d3= self.down3(pool2, training=training)
    pool3 = self.pool2(d3)
    d4= self.down4(pool3, training=training)
    pool4 = self.pool2(d4)


    u4 = self.up5(pool4, training=training)

    u4 = self.concat4([d4, u4])
    u3 = self.up4(u4, training=training)
    u3 = self.concat3([d3, u3])
    u2 = self.up3(u3, training=training)
    u2 = self.concat2([d2, u2])
    u1 = self.up2(u2, training=training)
    u1 = self.concat1([d1, u1])

    x = self.up1_conv0(u1, training=training)
    x = self.up1_conv1(x, training=training)
    x = self.up1_conv2(x)
    return x

  @staticmethod
  def build_keras_model():
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    x = tf.keras.layers.Input(shape=(320, 400, 3))
    y_hat = UNet()(x)

    model = tf.keras.Model(inputs=x, outputs=y_hat)
    model.compile(optimizer=optimizer, loss=Loss.Lyft_loss,
                              metrics=[Loss.Lyft_FScore, Loss.Lyft_car_Fscore, Loss.Lyft_road_Fscore])
    return model


if __name__ == '__main__':
    import tensorflow.contrib.eager as tfe
    import time

    tfe.enable_eager_execution()
    x = tf.random_normal((1, 320, 400, 3))
    model = UNet()
    iter = 1
    now = time.time()
    for i in range(iter):
      model(x)
    duration = (time.time() - now) / iter
    print("Time taken:", duration)