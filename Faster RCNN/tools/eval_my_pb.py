import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensorflow.python.framework import graph_util
output_graph_path = '../frozen_inference_graph.pb'
with tf.Session() as sess:
    with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        summary_write = tf.summary.FileWriter('./log', sess.graph)
        sess.run(summary_write)
# with open(output_graph_path, "rb") as f:
# output_graph_def.ParseFromString(f.read())
# _ = tf.import_graph_def(output_graph_def, name=""


# def restore_mode_pb( ):
#     sess = tf.Session()
#     with gfile.FastGFile('../frozen_inference_graph.pb', 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')
#         summary_write=tf.summary.FileWriter('./log',sess.graph)
#         sess.run(summary_write)
#     sess.close()
# restore_mode_pb()