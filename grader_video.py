import sys, skvideo.io, json, base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model_v4 import DilatedCNN as SegModel
from UNet import UNet as SegModel

# tfe.enable_eager_execution()

file = sys.argv[-1]

if file == 'demo.py':
  print("Error loading video")
  quit

model = SegModel.build_keras_model()
# model.load_weights("model_v4_weights.h5")
model.load_weights("UNet.h5")

video = skvideo.io.vread(file)
writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

answer_key = {}

ones = tf.ones((1, 600, 800, 3)) * 255
zeros = tf.zeros((1, 600, 800, 3))

for rgb_frame in video:
  resized = cv2.resize(rgb_frame, (400, 320))
  x = np.expand_dims(resized, axis=0)
  y_hat = model.predict_on_batch(x)[0]

  y_hat = np.where(y_hat >= 0.5, 255, 0).astype('uint8')
  y_hat = cv2.resize(y_hat, (800, 600))

  road = y_hat[:, :, 0]
  car = y_hat[:, :, 1]
  other = y_hat[:, :, 2]
  # other = rgb_frame[:, :, 2]

  overlay = np.stack([car, road, other], axis=2)
  frame = cv2.addWeighted(rgb_frame, 0.5, overlay, 0.5, 0)

  outimage = np.hstack((rgb_frame, frame))
  writer.writeFrame(outimage)


writer.close()
