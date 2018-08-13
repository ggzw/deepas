import functools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import random
import time
import numpy as np
import deep_AS_config

def lazy_property(function):
	attribute = '_' + function.__name__

	@property
	@functools.wraps(function)
	def wrapper(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)
	return wrapper

step = 0

class VariableSequenceClassification:

	def __init__(self, num_hidden=512, num_layers=5,dropout=1):
		self.data = tf.placeholder(tf.float32,
							shape = [None,deep_AS_config.FLAGS.maxlen,deep_AS_config.FLAGS.seq_len*6],
							name="LSTMinput")
		self.target = tf.placeholder(tf.float32,
							shape = [None,deep_AS_config.FLAGS.path_class],
							name="LSTMtarget")
		self._num_hidden = num_hidden
		self._num_layers = num_layers
		self.model_dir = '../'+ 'model/'
		self.log_dir = '../'+ 'log/'
		self.dropout = dropout
		self.prediction
		self.cost
		self.error
		self.optimize

	@lazy_property
	def length(self):
		used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
		length = tf.reduce_sum(used, reduction_indices=1)
		length = tf.cast(length, tf.int32)
		return length

	@lazy_property
	def prediction(self):
		# Recurrent network.
		#with tf.variable_scope('xavier',tf.contrib.layers.xavier_initializer()):
			
			#目前没有加dropout
			#network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=self.dropout)
			network = tf.contrib.rnn.LSTMCell(num_units=self._num_hidden)	
			#network = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(i) for  i in [512,256,128,64,32]])
			
			
			output, _ = tf.nn.dynamic_rnn(
				cell = network,
				inputs = self.data,
				dtype=tf.float32,
				sequence_length=self.length,
				)
			#不知道这个转至的意思
			#output = tf.transpose(output,[1, 0, 2])
			print(tf.shape(output))
			last = self._last_relevant(output, self.length)
			
			# Softmax layer.
			weight, bias = self._weight_and_bias(512,int(self.target.get_shape()[1]))
			prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
			return prediction

	@lazy_property
	def cost(self):
		cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
		return cross_entropy

	@lazy_property
	def optimize(self):
		learning_rate = 0.005 / deep_AS_config.FLAGS.current_epoch
		optimizer = tf.train.RMSPropOptimizer(learning_rate)
		return optimizer.minimize(self.cost)

	@lazy_property
	def error(self):
		mistakes = tf.equal(
			tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
		return tf.reduce_mean(tf.cast(mistakes, tf.float32))

	@staticmethod
	def _weight_and_bias(in_size, out_size):
		weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
		bias = tf.constant(0.1, shape=[out_size])
		return tf.Variable(weight), tf.Variable(bias)

	@staticmethod
	def _last_relevant(output, length):
		#print(output)
		batch_size = tf.shape(output)[0]
		max_length = tf.shape(output)[1]
		output_size = int(output.get_shape()[2])
		index = tf.range(0, batch_size) * max_length + (length - 1)
		flat = tf.reshape(output, [-1, output_size])
		relevant = tf.gather(flat, index)
		return relevant

	def restore_model(self, session):
		print("".join(["="] * 80)) # section-separating line
		print("Model: restore_model()")
		self.saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		global step
		if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
			print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
			step = int(ckpt.model_checkpoint_path.split('-')[1])
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
		input_feed={}
		input_feed[self.data] = input_x
		input_feed[self.target] = input_y
		_,cost,acc=sess.run([self.optimize,self.cost,self.error],feed_dict=input_feed)
		return cost,acc
	def teststep(self,sess,input_x):
		input_feed={}
		input_feed[self.data] = input_x
		#input_feed[self.target] = input_y
		pre =sess.run(self.prediction,feed_dict=input_feed)
		return pre

def database(dir, max_len):
	f = open(dir, 'r')
	decodelabel = {"0": [1, 0],"1": [0, 1]}
	decodex = {'N':[1,0,0,0,0,0],'A':[0,1,0,0,0,0],'T':[0,0,1,0,0,0],'C':[0,0,0,1,0,0],'G':[0,0,0,0,1,0],'0':[0,0,0,0,0,1]}
	x_one_hot = []
	y_one_hot = []
	print("#####读取data")
	data = f.readlines()
	print("#####over data")
	print("数据集的大小有：  ", len(data))

	for line in data:
		x = line.split()[:-1]
		#array = [decodex['0'] * 64] * max_len
		array = [[0,0,0,0,0,0] * 64] * max_len
		if len(x) > max_len:
			continue
		for i in range(len(x)):
			site = x[i].upper()
			exon = []
			for code in list(site):
				exon.append(decodex[code])
			array[i] = sum(exon,[])
		y = line.split()[-1]
		x_one_hot.append(array)
		y_one_hot.append(decodelabel[y])
	print("剪掉了",max_len,"以上之后还有 ：",len(y_one_hot))

		# xx = np.array(x_one_hot)
		# yy = np.array(y_one_hot)
	return x_one_hot, y_one_hot, len(y_one_hot)


def getbatch(x, y, lenth,batch_size = 64):
	x_batch = []
	y_batch = []
	#batch_size = 64
	for i in range(batch_size):
		j = random.randint(0, lenth - 1)
		x_batch.append(x[j])
		y_batch.append(y[j])
	#print(y_batch)
	return x_batch, y_batch


#	if __name__ == '__main__':
 	# We treat images as sequences of pixel rows.
 	#length = 300
# 	image_size = 384
# 	num_classes = 2
 	#dir = "../../../home/zhangcheng/alt/data64/protein_transgraph_test"

 	#x, y, l = database(dir, length)

# 	data = tf.placeholder(tf.float32, [None, length, image_size])
# 	target = tf.placeholder(tf.float32, [None, num_classes])
 	#model = VariableSequenceClassification()
 	#sess = tf.Session()
 	#sess.run(tf.global_variables_initializer())
 	#for epoch in range(10):
 	#	accs = 0
 	#	costs = 0
 	#	for _ in range(100):
 	#		batch_x,batch_y = getbatch(x,y,l)
 	#		_,cost,error = sess.run([model.optimize,model.cost,model.error],{data: batch_x, target: batch_y})
 	#		accs += error
 	#		costs += cost
 	#	t_cost = costs/100
 	#	t_acc = accs/100
 	#	print(t_cost)
 	#	print(t_acc)
 	#	batch_x, batch_y = getbatch(x, y, l)
 	#	acc = sess.run(model.error, {data: batch_x, target: batch_y})
 	#	print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
