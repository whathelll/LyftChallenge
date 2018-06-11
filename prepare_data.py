import cv2
import numpy as np
import glob
from generator import preprocess_labels
from sklearn.model_selection import train_test_split

def load_files(folder="Train", reduce=False):
  images = []
  masks = []
  for file in glob.glob("./" + folder + "/CameraRGB/*.png"):
      img = cv2.imread(file)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if reduce:
        img = cv2.resize(img, (400, 320))
      images.append(img)
  for file in glob.glob("./" + folder + "/CameraSeg/*.png"):
      img = cv2.imread(file)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = preprocess_labels(img)
      if reduce:
        img = cv2.resize(img, (400, 320))
      masks.append(img)
  return images, masks


def main():
  # images, masks = load_files()
  #
  # x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=0.2)
  # x_train = np.stack(x_train)
  # x_test = np.stack(x_test)
  # y_train = np.stack(y_train)
  # y_test = np.stack(y_test)
  #
  # np.save("x_train", x_train)
  # np.save("x_test", x_test)
  # np.save("y_train", y_train)
  # np.save("y_test", y_test)


  # images, masks = load_files(folder="Train", reduce=True)
  # x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=0.2)
  # x_train = np.stack(x_train)
  # y_train = np.stack(y_train)
  # np.save("x_train_s", x_train)
  # np.save("y_train_s", y_train)

  images, masks = load_files(folder="Test", reduce=True)
  x_test_s = np.stack(images)
  y_test_s = np.stack(masks)
  np.save("x_test_s", x_test_s)
  np.save("y_test_s", y_test_s)

if __name__ == '__main__':
    main()
