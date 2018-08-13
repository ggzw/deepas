import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import deep_AS_main_model
import deep_AS_config

def main(_):
    if deep_AS_config.FLAGS.model == "train1" or "train2" or "train3":
        deep_AS_main_model.train()
    elif deep_AS_config.FLAGS.model == "test1" or "test2" or "test3":
        deep_AS_main_model.test()
    elif deep_AS_config.FLAGS.model == "train4":
        deep_AS_main_model.LSTMtrain()
    elif deep_AS_config.FLAGS.model == "test4":
        deep_AS_main_model.testLSTM()
if __name__ == "__main__":
    tf.app.run()
