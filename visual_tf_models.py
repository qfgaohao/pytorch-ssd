import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import time

if len(sys.argv) < 2:
    print("Usage: python visual_tf_model.py <model.pb>")
    sys.exit(0)

model_file_name = sys.argv[1]
with tf.Session() as sess:
    with gfile.FastGFile(model_file_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='log'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

while True:
    time.sleep(1000)