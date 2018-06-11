import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from UNet import UNet as SegModel
from model_v4 import DilatedCNN as SegModel

import time
import numpy as np
import argparse
from modules import ConvModule
import tensorflow.contrib.eager as tfe
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model_save_file = "UNet_bak2.h5"

parser = argparse.ArgumentParser(description='testing freeze model')
parser.add_argument('--model_dir', type=str, default="./checkpoints/")
flags = parser.parse_args()

def freeze_model():
  feed = np.random.random((1, 320, 400, 3))

  with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, (None, 320, 400, 3))
    # model = ConvModule(64)
    model = SegModel()
    # model.load_weights(model_save_file)
    adam = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()

    checkpoint = tfe.Checkpoint(model=model, optimizer=adam, step_counter=global_step)
    logits = model(x)


    with tf.Session() as sess:
      # checkpoint.restore(tf.train.latest_checkpoint(flags.model_dir))#.assert_consumed().run_restore_ops()
      print("Testing")
      sess.run(tf.global_variables_initializer())
      y_result = sess.run(logits, feed_dict={x: feed})

      # test speed
      start = time.time()
      steps = 100
      for i in range(steps):
        y_result = sess.run(logits, feed_dict={x: feed})
      duration = (time.time() - start) / steps
      print("average duration:", round(duration, 4))

      print(y_result.shape)

      # for op in tf.get_default_graph().get_operations():
        # print(op.name)
      for input in model.inputs:
        print(input)

      for out in model.outputs:
        print(out.op.name)
      # print(model.output)

      # # Now, let's use the Tensorflow backend to get the TF graphdef and frozen graph
      # saver = tf.train.Saver()
      # # save model weights in TF checkpoint
      # checkpoint_path = saver.save(sess, "./models/snapshot", global_step=0, latest_filename='checkpoint_state')

      # train_graph = sess.graph
      # inference_graph = tf.graph_util.remove_training_nodes(train_graph.as_graph_def())
      #
      # graph_io.write_graph(inference_graph, '.', "./models/keras_graphdef.pb")


      # frozen_graph = freeze_session(sess, output_names=["u_net/conv2d/Reshape_1"])
      frozen_graph = freeze_session(sess, output_names=["dilated_cnn/conv2d_1/Reshape_1"])

      graph_io.write_graph(frozen_graph, '.', "./models/keras_frozen_model.pb", as_text=False)

  """This doesn't seem to work at the moment, so using freeze_session instead"""
  # freeze_graph.freeze_graph(
  #   "./models/keras_graphdef.pb",
  #   '',
  #   False,
  #   checkpoint_path,
  #   "dilated_cnn/conv2d_1/Reshape_1",
  #   "save/restore_all",
  #   "save/Const:0",
  #   "./models/keras_frozen_model.pb",
  #   False,
  #   ""
  # )


  print("Global_step:", global_step)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
  graph = session.graph
  with graph.as_default():
    freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
    output_names = output_names or []
    output_names += [v.op.name for v in tf.global_variables()]
    input_graph_def = graph.as_graph_def()
    if clear_devices:
      for node in input_graph_def.node:
        node.device = ""
    frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                  output_names, freeze_var_names)
    return frozen_graph


if __name__ == '__main__':
  freeze_model()