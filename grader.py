# python -W ignore grader.py

import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from model_v4 import DilatedCNN
import cv2

tfe.enable_eager_execution()

file = sys.argv[-1]

if file == 'grader.py':
  print("Error loading video")
  quit


def cv_encode(array):
  retval, buffer = cv2.imencode('.png', array)
  return base64.b64encode(buffer).decode("utf-8")

# Define encoder function
def encode(array):
  pil_img = Image.fromarray(array)
  buff = BytesIO()
  pil_img.save(buff, format="PNG")
  return base64.b64encode(buff.getvalue()).decode("utf-8")

model = DilatedCNN()
model.load()

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

ones = tf.ones((1, 600, 800, 3))
zeros = tf.zeros((1, 600, 800, 3))

for rgb_frame in video:
  # rgb_frame = cv2.resize(rgb_frame, (400, 320))
  x = tf.constant(np.expand_dims(rgb_frame, axis=0), dtype=tf.float32)
  x = tf.image.resize_images(x, (320, 400))

  y_hat = model(x)
  y_hat = tf.image.resize_images(y_hat, (600, 800))
  y_hat = tf.where(y_hat >= 0.5, ones, zeros)
  y_hat = tf.squeeze(y_hat).numpy().astype('uint8')

  road = y_hat[:, :, 0]
  car = y_hat[:, :, 1]
  answer_key[frame] = [cv_encode(car), cv_encode(road)]

  # Increment frame
  frame += 1


# Print output in proper json format
print(json.dumps(answer_key))
