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





if __name__ == "__main__":
  unittest.main()