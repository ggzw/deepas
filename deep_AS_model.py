import numpy as np
import tensorflow as tf
import deep_AS_config

import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class ModelNetwork(object):
    """
    creat the model tensor
    """
    def __init__(self):
        self.batch_size = deep_AS_config.FLAGS.batch_size
        self.seq_len = deep_AS_config.FLAGS.seq_len
        
        self.vocab_size = deep_AS_config.FLAGS.vocab_size
        self.num_units = deep_AS_config.FLAGS.num_units
        self.channel = 1
        #keep_prob probability of dropout layers, will be defined in build()
        self.dropout_keep = {}
        self.dropout_keep["conv"] = tf.placeholder(tf.float32,name = 'c_keep_prob')
        self.dropout_keep["dense"] = tf.placeholder(tf.float32, name='d_keep_prob')

    def _build_cnn(self,input_tensor,train_layer):
        """
        input :
        input_tensor:shape[batch_size,site_num,vocab_size]
        output :
        cnn : shape[batch_size,num_units]
        """
        print("".join(["="]*80))#section-separation line
        print("ModelNetwork: _build_cnn")
        #self.dropout_keep["dense"] = 0.9

        input_tensor = tf.reshape(input_tensor,
                                  [self.batch_size,self.seq_len,self.vocab_size,self.channel])


        #conv1:[128,64,5,1]>>[128,64,5,64]
        conv1_weight = tf.get_variable(
                                    name="conv1_weights", 
                                    shape=[3, 3, 1, 64],
                                    initializer=tf.uniform_unit_scaling_initializer(1.43))
        conv1_bias = tf.get_variable(
                                    name="conv1_biases",
                                    shape=[64],
                                    initializer=tf.constant_initializer(0.1))
        # y = relu[w * x +b]
        #strides 1,1,1,1 is the moving_step
        #padding 
        out = tf.nn.relu(tf.nn.conv2d(input_tensor,
                                        conv1_weight,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
                            +conv1_bias)
        #maxpooling [1,2,1,1] with stride [1,2,1,1]
        #so [128,64,5,64]>>[128,32,5,64]
        out = tf.nn.max_pool(out,
                           ksize=[1, 2, 1, 1],
                           strides=[1, 2, 1, 1],
                           padding='SAME')
        #conv = tf.nn.dropout(, self.dropout_keep["conv"])

        #conv2:[128,32,5,64]>>[128,32,5,128]
        conv2_weight = tf.get_variable(
                                    name="conv2_weights", 
                                    shape=[3, 3, 64, 128],
                                    initializer=tf.uniform_unit_scaling_initializer(1.43))
        conv2_bias = tf.get_variable(
                                    name="conv2_biases",
                                    shape=[128],
                                    initializer=tf.constant_initializer(0.1))

        out = tf.nn.relu(tf.nn.conv2d(out,
                                    conv2_weight,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                            +conv2_bias)
        #maxpooling [1,2,1,1] with stride [1,2,1,1]
        #so [128,32,5,128]>>[128,16,5,128]
        out = tf.nn.max_pool(out,
                           ksize=[1, 2, 1, 1],
                           strides=[1, 2, 1, 1],
                           padding='SAME')
        #conv = tf.nn.dropout(, self.dropout_keep["conv"])

        #conv3:[128,16,5,128]>>[128,16,5,256]
        conv3_weight = tf.get_variable(
                                    name="conv3_weights", 
                                    shape=[3, 3, 128,256],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv3_bias = tf.get_variable(
                                    name="conv3_biases",
                                    shape=[256],
                                    initializer=tf.constant_initializer(0.1))

        out = tf.nn.relu(tf.nn.conv2d(out,
                                    conv3_weight,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                            +conv3_bias)
        #maxpooling [1,2,1,1] with stride [1,2,1,1]
        #so [128,16,5,256]>>[128,8,5,256]
        out = tf.nn.max_pool(out,
                           ksize=[1, 2, 1, 1],
                           strides=[1, 2, 1, 1],
                           padding='SAME')
        #conv = tf.nn.dropout(, self.dropout_keep["conv"])


        #set dense size
        if train_layer==1:
            dense_input_size = 5 * 8 * 256
        elif train_layer==2:
            dense_input_size = 5 * 16 * 256

        dense_output_size = self.num_units
        dense1_weight = tf.get_variable(
                                        name="dense1_weights", 
                                        shape=[dense_input_size, dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense1_bias = tf.get_variable(
                                    name="dense1_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        dense1_input = tf.reshape(out, [-1, dense_input_size])
        out = tf.nn.relu(tf.matmul(dense1_input, dense1_weight) + dense1_bias)
        out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout1")

        #2dense
        dense_input_size = dense_output_size
        dense2_weight = tf.get_variable(
                                        name="dense2_weights", 
                                        shape=[dense_input_size, dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense2_bias = tf.get_variable(
                                    name="dense2_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        out = tf.nn.relu(tf.matmul(out, dense2_weight) + dense2_bias)
        out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout2")

        #3dense
        dense3_weight = tf.get_variable(
                                        name="dense3_weights", 
                                        shape=[dense_input_size, dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense3_bias = tf.get_variable(
                                    name="dense3_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        out = tf.nn.relu(tf.matmul(out, dense3_weight) + dense3_bias)
        out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout3")
        #4dense
        dense4_weight = tf.get_variable(
                                        name="dense4_weights", 
                                        shape=[dense_input_size, dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense4_bias = tf.get_variable(
                                    name="dense4_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        out = tf.nn.relu(tf.matmul(out, dense4_weight) + dense4_bias)
        out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout4")

        #5dense
        dense5_weight = tf.get_variable(
                                        name="dense5_weights", 
                                        shape=[dense_input_size, dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense5_bias = tf.get_variable(
                                    name="dense5_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        out = tf.nn.relu(tf.matmul(out, dense5_weight) + dense5_bias)
        out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout5")
        #6dense
        dense6_weight = tf.get_variable(
                                        name="dense6_weights", 
                                        shape=[dense_input_size, dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense6_bias = tf.get_variable(
                                    name="dense6_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        out = tf.nn.relu(tf.matmul(out, dense6_weight) + dense6_bias)
        out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout6")
        #7dense
        dense7_weight = tf.get_variable(
                                        name="dense7_weights", 
                                        shape=[dense_input_size, dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense7_bias = tf.get_variable(
                                    name="dense7_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        out = tf.nn.relu(tf.matmul(out, dense7_weight) + dense7_bias)
        out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout7")
        #8dense
        dense_output_size = deep_AS_config.FLAGS.label_class
        dense8_weight = tf.get_variable(
                                        name="dense8_weights", 
                                        shape=[dense_input_size,dense_output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        dense8_bias = tf.get_variable(
                                    name="dense8_biases", 
                                    shape=[dense_output_size],
                                    initializer=tf.constant_initializer(0.1))
        out = tf.nn.relu(tf.matmul(out, dense8_weight) + dense8_bias)
        #out = tf.nn.dropout(out, self.dropout_keep["dense"], name="dropout8")
        return out

    def get_logits(units_out):
	    #cnn_input = tf.reshape(x,[deep_AS_config.batch_size, deep_AS_config.num_units, deep_AS_config.vocab_size,1])
        #logit = tf.nn.relu(tf.matmul(out, dense8_weight) + dense8_bias)
	    logit= tf.softmax(tf.matmul(units_out, softmax_weight1) + softmax_bias1)
	    return logit




class Model(object):
  def __init__(self):
    print("".join(["="] * 80))
    print("Model: __init__()")
    
    #self.global_step = tf.get_variable(
    #                                  name="global_step",
    #                                  shape=[1],
    #                                  dtype="int32",
    #                                  initializer=tf.constant_initializer(1))
    # we currently use deepnovo_config to store both const & settings
    # the settings should be shown in __init__() to keep track carefully
    self.vocab_size = deep_AS_config.FLAGS.vocab_size
    self.num_units = deep_AS_config.FLAGS.num_units
    self.model_dir = deep_AS_config.FLAGS.model_dir
    self.batch_size = deep_AS_config.FLAGS.batch_size
    self.seq_len = deep_AS_config.FLAGS.seq_len
    self.label_class = deep_AS_config.FLAGS.label_class

    # input tensors are grouped into a dictionary
    self.input_dict = {}

    # input intensity profile: [batch_size, vocab_size, num_ion, WINDOW_SIZE]
    #   for example; [128, 64, 5]
    self.input_dict["data1_x"] = tf.placeholder(tf.float32,
                                                shape = [None,self.seq_len, self.vocab_size],
                                                name="x1_input")
    self.input_dict["data1_y"] = tf.placeholder(tf.int32,
                                                shape = [None,self.label_class],
                                                name="y1_input")
    ## input lstm state is a tuple of 2 tensors [batch_size, num_units]
    ##   for example: [128, 512]
    #self.input_dict["lstm_state"] = (tf.placeholder(dtype=tf.float32,
    #                                               shape=[None, self.num_units],
    #                                               name="input_c_state"), # to change to "input_lstm_state_c"
    #                                tf.placeholder(dtype=tf.float32,
    #                                               shape=[None, self.num_units],
    #                                               name="input_h_state")) # to change to "input_lstm_state_h"
    #
    # the keep_prob probability of dropout layers
    #   for inference model, they are const 1.0
    #   for train/valid model, they are input tensors
    self.dropout_keep = {}
    #self.dropout_keep["conv"] = 1.0
    #self.dropout_keep["dense"] = 1.0

    # core neural networks to calculate output tensors from the input
    self.model_network

    self.learning_rate = deep_AS_config.FLAGS.learning_rate

    self.train_op

    # output tensors are grouped into 2 dictionaries, forward and backward,
    #   each has 4 tensors:
    #   ["logit"]: shape [batch_size, vocab_size], to compute loss in training
    #   ["logprob"]: shape [batch_size, vocab_size], to compute score in inference
    #   ["lstm_state"]: shape [batch_size, num_units], to compute next iteration
    #   ["lstm_state0"]: shape [batch_size, num_units], state from cnn_spectrum
    # they will be built and loaded by build_model() and restore_model()
    self.output
    self.loss
    self.train_step
    self.accuracy
    #self.output_backward = None

    self.saver = None
  #@property
  #def global_step_plus(self):
  #  return self.global_step + 1
  @lazy_property
  def model_network(self):
    return ModelNetwork()
  @lazy_property
  def train_op(self):
    return tf.train.AdamOptimizer(self.learning_rate)
  @lazy_property
  def output(self):
    return self.model_network._build_cnn(self.input_dict["data1_x"],2)
  @lazy_property
  def loss(self):
     return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.input_dict["data1_y"]))
  @lazy_property
  def train_step(self):
    return self.train_op.minimize(self.loss)
  @lazy_property
  def accuracy(self):
    correct_pred = tf.equal(tf.argmax(self.output, 1),tf.argmax(self.input_dict["data1_y"], 1))
    return tf.reduce_mean(tf.cast(correct_pred,tf.float32))

  @lazy_property
  def build_model(self):
    print("".join(["="] * 80)) # section-separating line
    print("Model: build_model()")

  
  def restore_model(self, session):
    print("".join(["="] * 80)) # section-separating line
    print("Model: restore_model()")

    #self.build_model()


    self.saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
      print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
      self.saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      #~ train_writer = tf.train.SummaryWriter("train_log", session.graph)
      #~ train_writer.close()
      session.run(tf.global_variables_initializer())
  
  def save_model(self,session,checkpoint_path,step):
    self.saver.save(session,
                     checkpoint_path,
                     global_step=step,
                     write_meta_graph=False)
  
  def step(self,sess,input_x,input_y):
    input_feed = {}

    input_feed[self.input_dict["data1_x"]] = input_x
    input_feed[self.input_dict["data1_y"]] = input_y
    #input_x = np.reshape(input_x,[128,128,5])
    #print(np.shape(input_x))
    #if np.shape(input_x)==[128,128]:
    #    print(input_x)
    input_y = np.reshape(input_y,[128,2])
    input_feed[self.model_network.dropout_keep["dense"]] = 0.5
    input_feed[self.model_network.dropout_keep["conv"]] = 1.0
    _,loss,acc=sess.run([self.train_step,self.loss,self.accuracy],feed_dict=input_feed)
    return loss,acc

  
  def test_step(self,sess,input_x,input_y):
    input_feed = {}
    input_feed[self.input_dict["data1_x"]] = input_x
    input_feed[self.input_dict["data1_y"]] = input_y
    input_feed[self.model_network.dropout_keep["dense"]] = 1.0
    input_feed[self.model_network.dropout_keep["conv"]] = 1.0
    loss,acc=sess.run([self.loss,self.accuracy],feed_dict=input_feed)
    return loss,acc
