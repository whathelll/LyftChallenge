import tensorflow as tf

class Loss():
  @staticmethod
  def F_beta_score(y_true, y_pred, beta=1.):
    K = tf.keras.backend
    TP = K.sum(y_pred * y_true)
    FP = K.sum(y_pred * (1. - y_true))
    FN = K.sum((1. - y_pred) * y_true)

    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    return (1 + beta ** 2) * ((precision * recall) / (beta ** 2 * precision + recall + K.epsilon()))

  @staticmethod
  def Lyft_car_Fscore(y_true, y_pred):
    return Loss.F_beta_score(y_true[:, :, :, 1], y_pred[:, :, :, 1], beta=2.)

  @staticmethod
  def Lyft_road_Fscore(y_true, y_pred):
    return Loss.F_beta_score(y_true[:, :, :, 0], y_pred[:, :, :, 0], beta=0.5)

  @staticmethod
  def Lyft_FScore(y_true, y_pred):
    return (Loss.Lyft_car_Fscore(y_true, y_pred) + Loss.Lyft_road_Fscore(y_true, y_pred)) / 2.

  @staticmethod
  def Lyft_loss(y_true, y_pred):
    return 1 - Loss.Lyft_FScore(y_true, y_pred) #binary_crossentropy(y_true, y_pred) + 1 - Lyft_FScore(y_true, y_pred)