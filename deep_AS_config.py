import numpy as np
import tensorflow as tf

# ==============================================================================
# FLAGS
# ==============================================================================


tf.app.flags.DEFINE_string("train_dir", # flag_name
                           "../data64/splite_site_train_64bp",  # default_value
                           "Training directory.") # docstring

tf.app.flags.DEFINE_string("test_dir",
                           "../data64/splite_site_test_64bp",
                            "Testing directory")

tf.app.flags.DEFINE_integer("vocab_size",
                            5,
                            "vocab size the one hot length")
tf.app.flags.DEFINE_integer("batch_size",
                            8,
                            "Set the batch size.")
tf.app.flags.DEFINE_integer("num_units",
                            512,
                            "set the hidden size")
tf.app.flags.DEFINE_integer("seq_len",
                            64,
                            "the splice site len")

tf.app.flags.DEFINE_integer("epoch",
                            100,
                            "Set epoch number.")
tf.app.flags.DEFINE_integer("learning_rate",
                            0.00005,
                            "Set learning rate")
tf.app.flags.DEFINE_integer("label_class",
                            4,
                            "Set the prediction label class site.")
tf.app.flags.DEFINE_string("model",
                            "train1",
                            "Set model of the main")
tf.app.flags.DEFINE_boolean("reuse_model",
                            True,
                            "Set to True to reuse a model")
tf.app.flags.DEFINE_string("model_dir",
                            "../G_bysj/model1",
                            "model dir to save and reload model")
tf.app.flags.DEFINE_string("rnn_dir",
                            "../Z_bysj/",
                            "rnn_dir")
tf.app.flags.DEFINE_string("log_dir",
                            "../G_bysj/log1",
                            "log dir to save and reload model")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",
                            10,
                            "checkpoint")
tf.app.flags.DEFINE_string("GPU",
                            "/gpu:2",
                            "指定GPU")
tf.app.flags.DEFINE_integer("path_class",
                            2,
                            "pathclass")
tf.app.flags.DEFINE_integer("maxlen",
                            300,
                            "maxlen")
tf.app.flags.DEFINE_integer("current_epoch",1,"改变学习率") 
FLAGS = tf.app.flags.FLAGS
