import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

import argparse
from modules import ConvModule
import tensorflow.contrib.eager as tfe
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tfe.enable_eager_execution()

CONFIG = {
    # Where to save models
    "graphdef_file": "./models/keras_graphdef.pb",
    "frozen_model_file": "./models/keras_frozen_model.pb",
    "snapshot_dir": "./models/snapshot",
}

parser = argparse.ArgumentParser(description='testing freeze model')
parser.add_argument('--model_dir', type=str, default="./checkpoints/")
flags = parser.parse_args()

def freeze_model():
  x = tf.random_normal((1, 300, 400, 3))
  model = ConvModule(64, kernel_size=(3, 3))
  adam = tf.train.AdamOptimizer()
  checkpoint_prefix = os.path.join(flags.model_dir, 'ckpt')
  global_step = tf.train.get_or_create_global_step()


  y = model(x)

  print("y:", y.shape)

  checkpoint = tfe.Checkpoint(model=model, optimizer=adam, step_counter=global_step)
  checkpoint.restore(tf.train.latest_checkpoint(flags.model_dir))

  print("Global_step:", global_step)

  checkpoint.save(checkpoint_prefix)


if __name__ == '__main__':
  freeze_model()