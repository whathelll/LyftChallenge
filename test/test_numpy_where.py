import unittest
import numpy as np

class TestModel(unittest.TestCase):
  def setUp(self):
    pass

  def test_model(self):
    y = np.array([[1, 0, 1, 0],
                  [1, 0, 1, 1],
                  [1, 1, 0, 1],
                  [1, 0, 0, 1]
                 ])

    print(np.where(y > 0, 255, 0))

    frame1 = np.random.random((320, 400, 3))
    frame2 = np.random.random((320, 400, 3))

    a = np.hstack((frame1, frame2))
    print(a.shape)


if __name__ == "__main__":
  unittest.main()