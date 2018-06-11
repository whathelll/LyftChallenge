import tensorflow as tf
import tensorflow.contrib.eager as tfe
from google.protobuf import text_format

import argparse
import tensorflow.contrib.eager as tfe
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser(description='testing freeze model')
parser.add_argument('--model_dir', type=str, default="./checkpoints/")
flags = parser.parse_args()


def load_graph_for_transform(frozen_graph_filename):
  # We load the protobuf file from the disk and parse it to retrieve the
  # unserialized graph_def
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    # if FLAGS.input_binary:
    graph_def.ParseFromString(f.read())
    # else:
    # text_format.Merge(f.read(), graph_def)
  return graph_def


def load_graph(frozen_graph_filename):
  # We import the graph_def into a new Graph and returns it
  with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(load_graph_for_transform(frozen_graph_filename), name='')
  return graph




from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework       import graph_io

# Load the frozen graph
graph = load_graph_for_transform('./models/keras_frozen_model.pb')

# Transform it
input_names = ['Placeholder']
# output_names = ['u_net/conv2d/Reshape_1']
output_names = ['dilated_cnn/conv2d_1/Reshape_1']
transforms = ['strip_unused_nodes',
              'remove_nodes(op=Identity, op=CheckNumerics)',
              'fold_constants(ignore_errors=true)',
              'fold_batch_norms',
              'fold_old_batch_norms',
              # 'quantize_weights',
              # 'quantize_nodes'
             ]

G_opt = TransformGraph(graph, input_names, output_names, transforms)

# Write it to disk
with tf.gfile.GFile('./models/keras_opt_model.pb', "wb") as f:
    f.write(G_opt.SerializeToString())

#Compare the number of operation before and after
graph = load_graph('./models/keras_frozen_model.pb')
print(len(graph.get_operations()))
# for op in graph.get_operations():
#    print(op.name)

graph = load_graph('./models/keras_opt_model.pb')
print(len(graph.get_operations()))
# for op in graph.get_operations():
#    print(op.name)