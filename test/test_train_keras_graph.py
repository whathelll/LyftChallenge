from UNet import UNet
from loss import Loss
import numpy as np
import tensorflow as tf
import cv2
import glob
from generator import preprocess_labels
import tensorflow.contrib.eager as tfe
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def load_test_data():
  image_files = []
  for file in glob.glob("./Train/CameraRGB/*.png"):
    image_files.append(file)
  mask_files = []
  for file in glob.glob("./Train/CameraSeg/*.png"):
    mask_files.append(file)

  images = []
  masks = []
  for i in range(0, 2):
    img = cv2.imread(image_files[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 320))
    images.append(img)

    mask = cv2.imread(mask_files[i])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = preprocess_labels(mask)
    mask = cv2.resize(mask, (400, 320))
    masks.append(mask)
  return images, masks


weight_file = "test_train_keras.h5"
model_file = "test_train_keras_model.h5"
images, labels = load_test_data()

x = tf.placeholder(tf.float32, (None, 320, 400, 3))
y = tf.placeholder(tf.float32, (None, 320, 400, 3))

model = UNet()
y_hat = model(x)
# optimizer = tf.keras.optimizers.Adam(lr=1e-5)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
loss = Loss.Lyft_loss(y, y_hat)
fscore = Loss.Lyft_FScore(y, y_hat)
car_fscore = Loss.Lyft_car_Fscore(y, y_hat)
road_fscore = Loss.Lyft_road_Fscore(y, y_hat)
global_step = tf.train.get_or_create_global_step()
train_op = optimizer.minimize(loss, global_step=global_step)

saver = tf.train.Saver([model, optimizer, global_step], max_to_keep=1)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # model.load_weights(weight_file)

  for i in range(30):
    _, train_loss, train_fscore, train_car_fscore, train_road_fscore = \
      sess.run([train_op, loss, fscore, car_fscore, road_fscore], feed_dict={x: images, y:labels})
    print("loss: {:.4f} - Lyft_FScore: {:.4f} - Lyft_car_Fscore: {:.4f} - Lyft_road_Fscore: {:.4f}".
          format(train_loss, train_fscore, train_car_fscore, train_road_fscore))

  saver.save(sess, "test_train_keras_model", global_step=global_step)
  # model.save_weights(weight_file)

  # _keras_model.compile(optimizer=optimizer, loss=Loss.Lyft_loss,
  #                      metrics=[Loss.Lyft_FScore, Loss.Lyft_car_Fscore, Loss.Lyft_road_Fscore])


"""Conclusion: use checkpoints, use tf.Train.Saver instead."""
print("Done")


