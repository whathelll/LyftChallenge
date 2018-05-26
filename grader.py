import sys, skvideo.io, json, base64
import numpy as np
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

for rgb_frame in video:
  x = tf.constant(np.expand_dims(rgb_frame, axis=0), dtype=tf.float32)
  y_hat = tf.squeeze(model(x)).numpy()
  y_hat = np.where(y_hat >= 0.5, 1, 0).astype('uint8')
  road = y_hat[:, :, 0]
  car = y_hat[:, :, 1]
  answer_key[frame] = [encode(car), encode(road)]

  # Increment frame
  frame += 1


# Print output in proper json format
print(json.dumps(answer_key))