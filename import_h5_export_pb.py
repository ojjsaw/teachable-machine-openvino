import os, sys
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

load_model_path=sys.argv[1]
save_pb_dir='.' # curr. working directory
save_pb_name=sys.argv[2]

tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(0) 
model = load_model(load_model_path)
session = tf.keras.backend.get_session()
with session.graph.as_default():
    graphdef_inf = tf.graph_util.remove_training_nodes(session.graph.as_graph_def())
    graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, [out.op.name for out in model.outputs])
    graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=False)
