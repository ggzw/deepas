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

tf.app.flags.DEFINE_integer("vocab_size",   #碱基类型
                            5,
                            "vocab size the one hot length")
tf.app.flags.DEFINE_integer("batch_size",
                            128,
                            "Set the batch size.")
tf.app.flags.DEFINE_integer("num_units",   #隐藏单元数
                            512,
                            "set the hidden size")
tf.app.flags.DEFINE_integer("seq_len",     #以剪接位点为中心的序列长度，代表一个剪接位点，两个剪接位点代表一个外显子或相邻外显子。
                            64,
                            "the splice site len")

tf.app.flags.DEFINE_integer("epoch",
                            100,
                            "Set epoch number.")
tf.app.flags.DEFINE_integer("learning_rate",
                            0.00005,
                            "Set learning rate")
tf.app.flags.DEFINE_integer("label_class",      #预测结果的种类
                            2,
                            "Set the prediction label class site.")
tf.app.flags.DEFINE_string("model",         #模型选择
                            "train1",
                            "Set model of the main")
tf.app.flags.DEFINE_boolean("reuse_model",     
                            True,
                            "Set to True to reuse a model")
tf.app.flags.DEFINE_string("model_dir",     
                            "../G_bysj/model1",
                            "model dir to save and reload model")
tf.app.flags.DEFINE_string("log_dir",
                            "../G_bysj/log1",
                            "log dir to save and reload model")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",   #没训练几步输出一次结果
                            10,
                            "checkpoint")
tf.app.flags.DEFINE_string("GPU",
                            "/gpu:2",
                            "指定GPU")
tf.app.flags.DEFINE_integer("path_class",
                            2,
                            "pathclass")
tf.app.flags.DEFINE_integer("maxlen",     #转录本含几个外显子*2（剪接位点个数）
                            300,
                            "maxlen")
tf.app.flags.DEFINE_integer("current_epoch",1,"改变学习率") 
FLAGS = tf.app.flags.FLAGS
