import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import scipy.ndimage as ndi
from replay_memory import PrioritisedReplayMemory

class MultiPlotter(object):
  def __init__(self, columns=3, figsize=(20,20)):
    self.columns = columns
    self.images = []
    self.figsize= figsize

  def append(self, image):
    self.images.append(image)
    return self

  def reset(self):
    self.images = []

  def show(self):
    plt.figure(figsize=self.figsize)
    count = len(self.images)
    rows = math.ceil(1.0 * count / self.columns)
    for i in range(count):
      plt.subplot(rows, self.columns, i + 1)
      plt.imshow(self.images[i])

    plt.show()


def multi_plot(images):
  plt.figure(figsize=(20, 20))
  count = len(images)

  for i in range(count):
    plt.subplot(1, count, i + 1)
    plt.imshow(images[i])
  plt.show()

def preprocess_labels(label_image):
  road = np.zeros_like(label_image[:, :, 0])
  lane_marking_pixels = (label_image[:, :, 0] == 6).nonzero()
  road_pixels = (label_image[:, :, 0] == 7).nonzero()
  road[lane_marking_pixels] = 1
  road[road_pixels] = 1

  car = np.zeros_like(road)
  vehicle_pixels = (label_image[:, :, 0] == 10).nonzero()
  # Isolate vehicle pixels associated with the hood (y-position > 496)
  hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
  hood_pixels = (vehicle_pixels[0][hood_indices], \
                 vehicle_pixels[1][hood_indices])
  car[vehicle_pixels] = 1
  car[hood_pixels] = 0

  others = np.zeros_like(road)
  others[(car != 1) & (road != 1)] = 1
  return np.stack([road, car, others], axis=2)



def trans_image(image, mask):
  # Translation
  scale = np.random.normal(loc=0.0, scale=.5)

  tx = 0
  ty = 100 * scale

  translation_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
  #     image_tr = cv.warpAffine(image,translation_matrix,(image.shape[1],image.shape[0]))
  image_tr = apply_transform(image, translation_matrix)
  mask_tr = apply_transform(mask, translation_matrix)
  return image_tr, mask_tr


def apply_transform(x,
                    transform_matrix,
                    channel_axis=2,
                    fill_mode='nearest',
                    cval=0.):
  """Apply the image transformation specified by a matrix.
  # Arguments
      x: 2D numpy array, single image.
      transform_matrix: Numpy array specifying the geometric transformation.
      channel_axis: Index of axis for channels in the input tensor.
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
      cval: Value used for points outside the boundaries
          of the input if `mode='constant'`.
  # Returns
      The transformed version of the input.
  """
  x = np.rollaxis(x, channel_axis, 0)
  final_affine_matrix = transform_matrix[:2, :2]
  final_offset = transform_matrix[:2, 2]
  channel_images = [ndi.interpolation.affine_transform(
    x_channel,
    final_affine_matrix,
    final_offset,
    order=0,
    mode=fill_mode,
    cval=cval) for x_channel in x]
  x = np.stack(channel_images, axis=0)
  x = np.rollaxis(x, 0, channel_axis + 1)
  return x


def augment_brightness_camera_images(image):
  image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  random_bright = .25 + np.random.uniform()
  toobright = image1[:, :, 2] > 255 / random_bright
  image1[:, :, 2] = image1[:, :, 2] * random_bright
  image1[:, :, 2][toobright] = 255
  image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
  return image1


def preprocess_image(x, y):
  # x, y = trans_image(x, y)

  # randomly adjust brightness
  x = augment_brightness_camera_images(x)

  # random flip
  flip = random.randint(0, 1)
  if flip == 1:
    x = np.fliplr(x)
    y = np.fliplr(y)

  return x, y


def generator(images, masks, memory, batch_size=256):
  while 1:  # Loop forever so the generator never terminates
    new_batch_size = batch_size
    if(len(memory) > batch_size):
      new_batch_size = round(batch_size/2)

    x = []
    y = []
    for i in range(new_batch_size):
      index = np.random.randint(len(images))
      image, mask = preprocess_image(images[index], masks[index])
      memory.push(image, mask)
      x.append(image)
      y.append(mask)

    x, y, indices = memory.sample(batch_size)
    # yield np.stack(x), np.stack(y)
    yield x, y, indices