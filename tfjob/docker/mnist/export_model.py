"""Export given TensorFlow model.
The model is a pretrained  "MNIST", which saved as TensorFlow model checkpoint. This program
simply uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_export.py [--model_version=y] [--checkpoint_dir=checkpoint_oss_path] [--checkpoint_step=checkpoint_step] export_dir
"""

import os
import sys

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the exported model.')
tf.app.flags.DEFINE_integer('checkpoint_step', 0, 'Checkpoint steps that we export.')
tf.app.flags.DEFINE_string('checkpoint_path', None, 'Checkpoints path.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_dist_export.py '
          '[--model_version=y] [--checkpoint_path=checkpoint_store_path] [--checkpoint_step=checkpoint_step] export_dir')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print('Please specify a positive value for exported serveable version number.')
    sys.exit(-1)
  if not FLAGS.checkpoint_path:
    print('Please specify the correct path where checkpoints stored locally or in OSS.')
    sys.exit(-1)
  
  checkpoint_basename="model.ckpt"
  default_meta_graph_suffix='.meta'
  ckpt_path=os.path.join(FLAGS.checkpoint_path, checkpoint_basename + '-' + str(FLAGS.checkpoint_step))
  meta_graph_file=ckpt_path + default_meta_graph_suffix
  with tf.Session() as new_sess:
#   with new_sess.graph.as_default():
  #  tf.reset_default_graph()
  #  new_sess.run(tf.initialize_all_variables())
    new_saver = tf.train.import_meta_graph(meta_graph_file, clear_devices=True) #'/test/mnistoutput/ckpt.meta')
    new_saver.restore(new_sess, ckpt_path) #'/test/mnistoutput/ckpt')
    new_graph = tf.get_default_graph()
    new_x = new_graph.get_tensor_by_name('input/x-input:0')
    print(new_x)
    new_y = new_graph.get_tensor_by_name('layer2/activation:0')
    print(new_y)

  # Export model
  # WARNING(break-tutorial-inline-code): The following code snippet is
  # in-lined in tutorials, please update tutorial documents accordingly
  # whenever code changes.
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
    tensor_info_x = utils.build_tensor_info(new_x)
    tensor_info_y = utils.build_tensor_info(new_y)

    prediction_signature = signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
      new_sess, [tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
      },
      legacy_init_op=legacy_init_op,
      clear_devices=True)
    builder.save()

  print('Done exporting!')

if __name__ == '__main__':
  tf.app.run()