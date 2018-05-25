from collections import namedtuple, deque
import random
import numpy as np

Sample = namedtuple("Sample", ["x", "y"])

class PrioritisedReplayMemory(object):
  def __init__(self, capacity=1000, e=0.1, alpha=0.5):
    self.capacity = capacity
    self.memory = []
    self.position = 0
    self.errors = []
    self.e = e
    self.alpha = alpha
    self.beta = 1

  def push(self, x, y):
    if len(self.memory) < self.capacity:
      self.errors.append(None)
      self.memory.append(None)
    self.memory[self.position] = Sample(x, y)
    self.errors[self.position] = 10000
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size=2):
    probability = np.array(self.errors) ** self.alpha
    probability = probability / np.sum(probability)
    indices = np.random.choice(len(self.errors), size=batch_size, replace=False, p=probability)
    # samples = np.take(self.memory, indices, axis=0)
    x = []
    y = []
    for index in indices:
      x.append(self.memory[index][0])
      y.append(self.memory[index][1])

    return np.stack(x), np.stack(y), indices

  def update(self, indices, errors):
    errors = np.absolute(errors)
    for i in range(len(indices)):
      self.errors[indices[i]] = errors[i] + self.e


  def __len__(self):
    return len(self.memory)

  def __str__(self):
    result = []
    for i in range(self.__len__()):
      result.append("error:" + self.errors[i].__str__() + " \n")
    return "".join(result)





