import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tf.reset_default_graph()

sess = tf.Session()

# saver = tf.train.import_meta_graph('/root/models/optimize_me/linear/cpu/model.ckpt.meta')
# saver.restore(sess, '/root/models/optimize_me/linear/cpu/model.ckpt')

# optimize_me_parent_path = '/root/models/optimize_me/linear/cpu'
fully_optimized_frozen_model_graph_path = "./models/keras_opt_model.pb"
print(fully_optimized_frozen_model_graph_path)

with tf.gfile.GFile(fully_optimized_frozen_model_graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

tf.import_graph_def(
    graph_def,
    input_map=None,
    return_elements=None,
    name="",
    op_dict=None,
    producer_op_list=None
)

graph = sess.graph
print(len(graph.get_operations()))
# for op in graph.get_operations():
#    print(op.name)

# print("weights = ", sess.run("weights:0"))
# print("bias = ", sess.run("bias:0"))
x = np.random.random((1, 320, 400, 3))

# output_name = "u_net/conv2d/Reshape_1:0"
output_name = "dilated_cnn/conv2d_1/Reshape_1:0"
y = sess.run(output_name, feed_dict={"Placeholder:0": x})

start = time.time()
steps = 100
for i in range(steps):
  y = sess.run(output_name, feed_dict={"Placeholder:0": x})

duration = (time.time() - start) / steps
print("average duration:", round(duration, 4))
print(y.shape)


