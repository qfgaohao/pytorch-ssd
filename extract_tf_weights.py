import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import sys
import pickle


def read_weights(frozen_model):
    weights = {}
    with tf.Session() as sess:
        with gfile.FastGFile(frozen_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
        for n in graph_def.node:
            if n.op == 'Const':
                weights[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)
                print("Name:", n.name, "Shape:", weights[n.name].shape)
    return weights


if len(sys.argv)  < 3:
    print("Usage: python extract_tf_weights.py <frozen_model.pb> <weights_file.pickle>")

frozen_model = sys.argv[1]
weights_file = sys.argv[2]

weights = read_weights(frozen_model)
with open(weights_file, "wb") as f:
    pickle.dump(weights, f)
    print(f"Saved weights to {weights_file}.")