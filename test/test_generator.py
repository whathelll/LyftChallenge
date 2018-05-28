import unittest
import cv2
from generator import generator, preprocess_image, preprocess_labels, multi_plot, MultiPlotter
from replay_memory import PrioritisedReplayMemory
import numpy as np
import glob
import matplotlib.pyplot as plt




def load_test_data():
  image_files = []
  for file in glob.glob("./Train/CameraRGB/*.png"):
    image_files.append(file)
  mask_files = []
  for file in glob.glob("./Train/CameraSeg/*.png"):
    mask_files.append(file)

  images = []
  masks = []
  for i in range(0, 5):
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

class TestGenerator(unittest.TestCase):
  def setUp(self):
    images, masks = load_test_data()
    self.images = images
    self.masks = masks
    index = 0
    plotter = MultiPlotter(columns=4)
    plotter.append(images[index])\
      .append(masks[index][:, :, 0])\
      .append(masks[index][:, :, 1])\
      .append(masks[index][:, :, 2])
    plotter.show()

  def test_generate(self):
    memory = PrioritisedReplayMemory(capacity=2 * 10)
    gen = generator(self.images, self.masks, memory, batch_size=2)
    x, y, indices = next(gen)
    # self.assertEqual(x.shape, (2, 600, 800, 3))
    # self.assertEqual(y.shape, (2, 600, 800, 2))
    memory.update(indices, [0.1, 0.1])
    print(indices, memory.errors)
    # index = 0
    # multi_plot([x[index], y[index][:, :, 0], y[index][:, :, 1]])
    # index = 1
    # multi_plot([x[index], y[index][:, :, 0], y[index][:, :, 1]])

    x, y, indices = next(gen)
    print(indices, memory.errors)

    x, y, indices = next(gen)
    print(indices, memory.errors)

    x, y, indices = next(gen)
    print(indices, memory.errors)
    # index = 0
    # multi_plot([x[index], y[index][:, :, 0], y[index][:, :, 1]])
    # index = 1
    # multi_plot([x[index], y[index][:, :, 0], y[index][:, :, 1]])

  def test_multiplot(self):
    plotter = MultiPlotter(columns=4)

    # plotter.reset()
    images = self.images
    masks = self.masks
    for index in range(1):
      plotter.append(images[index])\
        .append(masks[index][:, :, 0])\
        .append(masks[index][:, :, 1])\
        .append(masks[index][:, :, 2])
    plotter.show()



if __name__ == "__main__":
  unittest.main()