import sys, skvideo.io, json, base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model import DilatedCNN

tfe.enable_eager_execution()

file = sys.argv[-1]

if file == 'demo.py':
  print("Error loading video")
  quit

model = DilatedCNN()
model.load()

video = skvideo.io.vread(file)
writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

answer_key = {}

for rgb_frame in video:
  resized = cv2.resize(rgb_frame, (400, 320))
  x = tf.constant(np.expand_dims(resized, axis=0), dtype=tf.float32)
  y_hat = tf.squeeze(model(x)).numpy()


  y_hat = np.where(y_hat >= 0.5, 255, 0).astype('uint8')
  y_hat = cv2.resize(y_hat, (800, 600))

  road = y_hat[:, :, 0]
  car = y_hat[:, :, 1]

  blank = np.zeros_like(road)
  overlay = np.stack([car, road, blank], axis=2)
  frame = cv2.addWeighted(rgb_frame, 0.5, overlay, 0.5, 0)

  writer.writeFrame(frame)


writer.close()
