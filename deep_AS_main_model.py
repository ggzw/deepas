#该文件主要为数据处理及训练测试
import tensorflow as tf
import deep_AS_config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import deep_AS_model
import LSTMclass as L
import random
import time

global_step = tf.get_variable('global_step', dtype='int32',shape=[1],initializer=tf.constant_initializer(0), trainable=False)
step = 0

def database(dir):              #预测剪接位点模型数据处理，每个碱基one-hot编码。
	f=open(dir,'r')
	decodedic = {'A':[0,0,0,1,0],
				'T':[0,0,1,0,0],
				'C':[0,1,0,0,0],
				'G':[1,0,0,0,0],
				'N':[0,0,0,0,1]}
	decodelabel = {"0":[1,0,0,0],
					"1":[0,1,0,0],
					"2":[0,0,1,0],
					"3":[0,0,0,1]}
	x_one_hot=[]
	y_one_hot=[]
	data = f.readlines()
	print("训练集的大小有：  ",len(data))
	for line in data:
		arrayx=[]
		x = line.split()[0]
		y = line.split()[2]
		for i in x:
			arrayx.append(decodedic[i])
		
		x_one_hot.append(arrayx)
		y_one_hot.append(decodelabel[y])
	return x_one_hot,y_one_hot,len(x_one_hot)

def getbatch(x,y,lenth):          #小批次训练，这里产生大小为batch_size的小批次。
	x_batch=[]
	y_batch=[]
	for i in range(deep_AS_config.FLAGS.batch_size):
		j=random.randint(0,lenth-1)
		x_batch.append(x[j])
		y_batch.append(y[j])
	return x_batch,y_batch

def database2(dir):      #外显子预测模型数据处理，处理碱基序列还有外显子长度作为输入，one-hot编码。
	f=open(dir,'r')
	decodedic = {'A':[0,0,0,1,0],
				'T':[0,0,1,0,0],
				'C':[0,1,0,0,0],
				'G':[1,0,0,0,0],
				'N':[0,0,0,0,1]}
	decodelabel = {"0":[1,0],
					"1":[0,1],}

	x_one_hot=[]
	y_one_hot=[]
	data = f.readlines()
	print("训练集的大小有：  ",len(data))
	for line in data:
		#print (len(line.split()))
		arrayx=[]
		x = line.split()[0]+line.split()[1]
		x = x.upper()
		y = line.split()[4]
		exon_length = int(line.split()[5])
		if 0 < exon_length <=100:
			exon_length = [1,0,0,0,0]
		elif 100 < exon_length <=500:
			exon_length = [0,1,0,0,0]
		elif 500 < exon_length <=1000:
			exon_length = [0,0,1,0,0]
		elif 1000 < exon_length <=2000:
 			exon_length = [0,0,0,1,0]
		elif exon_length > 2000:
 			exon_length = [0,0,0,0,1]
		for i in x:
			if decodedic.__contains__(i):
				a = decodedic[i]
				assert len(a) == 5
				arrayx.append(a)
			else:
				print(i)
				next
		arrayx[-1] = exon_length
		x_one_hot.append(arrayx)
		y_one_hot.append(decodelabel[y])
	return x_one_hot,y_one_hot,len(x_one_hot)

def train():     #预测剪接位点模型、外显子模型、相邻外显子模型的训练
	print("train()")
	# TRAINING on train_set
	if deep_AS_config.FLAGS.model == "train1":
		log_file = deep_AS_config.FLAGS.log_dir + "/log_file_data.tab"
		x,y,l = database(deep_AS_config.FLAGS.train_dir)
		print("database done")
		x_test, y_test, l_test = database(deep_AS_config.FLAGS.test_dir)
		print("test database done")
		model_dir = deep_AS_config.FLAGS.model_dir

	elif deep_AS_config.FLAGS.model == "train2":
		log_file = deep_AS_config.FLAGS.log_dir2 + "/log_file_data.tab"
		x,y,l = database2(deep_AS_config.FLAGS.train_dir2)
		print("database done")
		x_test, y_test, l_test = database2(deep_AS_config.FLAGS.test_dir2)
		print("test database done")
		model_dir = deep_AS_config.FLAGS.model_dir2
		deep_AS_config.FLAGS.seq_len = 128
		deep_AS_config.FLAGS.label_class = 2

	elif deep_AS_config.FLAGS.model == "train3":
		log_file = deep_AS_config.FLAGS.log_dir3 + "/log_file_data.tab"
		x,y,l = database3(deep_AS_config.FLAGS.train_dir3)
		print("database done")
		x_test, y_test, l_test = database3(deep_AS_config.FLAGS.test_dir3)
		print("test database done")
		model_dir = deep_AS_config.FLAGS.model_dir3
		deep_AS_config.FLAGS.seq_len = 128
		deep_AS_config.FLAGS.label_class = 2

	else :
		print("error")
		
	print("Open log_file: ", log_file)
	log_file_handle = open(log_file, 'w')
	print("global step \t epoch \t accurancy\t"
		"losses\n",
			file=log_file_handle,
			end="")

	global step
	with tf.Session() as sess:
		print("Create model for training")
		model = create_model(sess)
		step = sess.run(global_step)
		
#		layer1 = ["train1","test1"]
#		if deep_AS_config.FLAGS.model not in layer1:
#			model.seq_len = deep_AS_config.FLAGS.seq_len2
#			model.label_class = deep_AS_config.FLAGS.label_class2
		checkpoint_path = os.path.join(model_dir, "data.ckpt")
#		graphs_path = os.path.join(model_dir, "/graphs_layer2")
#		writer = tf.summary.FileWriter(graphs_path, sess.graph)
		epoch = 0
		while True:
			#规定训练轮数 大于训练轮数便停止
			epoch_last = epoch
			epoch = (step
			* deep_AS_config.FLAGS.batch_size
			//l)
			train_cycle(model,
						sess,
						epoch,
						checkpoint_path,
						x,
						y,
						l,
						log_file_handle)

			if epoch >= deep_AS_config.FLAGS.epoch:
				print("EPOCH: {0:.1f}, EXCEED {1:d}, STOP TRAINING LOOP".format(
					epoch,
					deep_AS_config.FLAGS.epoch))
				break
			iftest = epoch-epoch_last
			if iftest:
				print("".join(["="] * 80),file=log_file_handle,end="")
				print("have a test on test-data",file=log_file_handle,end="")
				print("".join(["="] * 80))  # section-separation line
				print("have a test on test-data")
				test_cycle(model,sess,x_test,y_test,l_test,log_file_handle)
		#关闭句柄
		log_file_handle.close()

def database3(dir):           #外显子相邻模型数据处理
	f=open(dir,'r')
	decodedic = {'A':[0,0,0,1,0],
				'T':[0,0,1,0,0],
				'C':[0,1,0,0,0],
				'G':[1,0,0,0,0],
				'N':[0,0,0,0,1]}
	decodelabel = {"0":[1,0],
					"1":[0,1],}

	x_one_hot=[]
	y_one_hot=[]
	data = f.readlines()
	print("训练集的大小有：  ",len(data))
	for line in data:
		#print (len(line.split()))
		arrayx=[]
		if line.split()[2]=="@":       #如果是第一个外显子，则左端添加32个N
			start="NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"+line.split()[0]
		else:
			start=line.split()[0]
		if line.split()[3]=="#":         #如果是最后一个外显子，则优端添加32个N
			end=line.split()[1]+"NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
		else:
			end=line.split()[1]
		x = start+end
		x = x.upper()
#		print(x)
		y = line.split()[4]
		for i in x:
			arrayx.append(decodedic[i])
		x_one_hot.append(arrayx)
		y_one_hot.append(decodelabel[y])
	return x_one_hot,y_one_hot,len(x_one_hot)

def create_model(session):

	print("create_model()")

	model = deep_AS_model.Model()
	model.restore_model(session)


	return model

def create_RNN_model(session):
	print("create_RNN_model()")
	
	rnn_model = L.VariableSequenceClassification()
	rnn_model.restore_model(session)
	return rnn_model

def train_cycle(model,
				sess,
				epoch,
				checkpoint_path,
				x,
				y,
				l,
				log_file_handle):
	# 调用model step
	# encoder_input >> step_loss output logists
	global step
	current_step = 0
	training_loss = 0
	acc_total = 0
	start = time.clock()
	while True:
		input_x,input_y = getbatch(x,y,l)
	#	print (np.shape(input_x))
		if deep_AS_config.FLAGS.model=="train1":
		#	print("start")
			step_loss,acc = model.step(sess,input_x,input_y)
			step +=1
			training_loss +=step_loss
			acc_total += acc
			current_step+=1
			if current_step % deep_AS_config.FLAGS.steps_per_checkpoint == 0:
				print ("%d \t %d \t %.4f \t %.4f \n"
		 			% ( step,
						epoch,
						acc_total,
						training_loss),
					file=log_file_handle,
					end="")
				log_file_handle.flush()
				end = time.clock()
				cycletime  = end - start
				print("step\t",step)
				print("损失\t", training_loss / deep_AS_config.FLAGS.steps_per_checkpoint)
				print("准确率\t", acc_total / deep_AS_config.FLAGS.steps_per_checkpoint)
				print ("完成了一个cycle","time:",cycletime)
				break
		elif deep_AS_config.FLAGS.model=="train2" or "train3":
			if np.shape(input_x)==(128, 128, 5):
			#	print("start")
				step_loss,acc = model.step(sess,input_x,input_y)
				step +=1
				training_loss +=step_loss
				acc_total += acc
				current_step+=1
				if current_step % deep_AS_config.FLAGS.steps_per_checkpoint == 0:
					print ("%d \t %d \t %.4f \t %.4f \n"
						% ( step,
							epoch,
							acc_total,
							training_loss),
						file=log_file_handle,
						end="")
					log_file_handle.flush()
					end = time.clock()
					cycletime  = end - start
					print("step\t",step)
					print("损失\t", training_loss / deep_AS_config.FLAGS.steps_per_checkpoint)
					print("准确率\t", acc_total / deep_AS_config.FLAGS.steps_per_checkpoint)
					print ("完成了一个cycle","time:",cycletime)
					break
			else:
				print("error shape")
		else:
			print("error label")		
	global_step = tf.Variable(step,name='global_step')
	model.save_model(sess,checkpoint_path,int(step))

def test_cycle(model,sess,x_test,y_test,l,log_file_handle):
	loss_t=0
	acc_t = 0
	input_x=[]
	input_y=[]
	number= 5 
	for n in range(number):
		for i in range(128):
			j=random.randint(0,l-1)
			input_x.append(x_test[j])
			input_y.append(y_test[j])
		loss, acc = model.test_step(sess, input_x, input_y)
		acc_t += acc
		loss_t += loss
		input_x = []	
		input_y = []
	print("测试——准确率\t", acc_t/number)
	print("测试——损失\t", loss_t/number)
	print("%.4f \t %.4f \n" % (acc_t/number,loss_t/number), file=log_file_handle,end="")
	log_file_handle.flush()

def test():
	
	print("make test database")
	if deep_AS_config.FLAGS.model == "test1":
		x,y,l = database(deep_AS_config.FLAGS.test_dir)
		restore_model_file = deep_AS_config.FLAGS.model_dir
	elif deep_AS_config.FLAGS.model == "test2":
		x,y,l = database2(deep_AS_config.FLAGS.test_dir2)
		restore_model_file = deep_AS_config.FLAGS.model_dir2
	elif deep_AS_config.FLAGS.model == "test3":
		x,y,l = database3(deep_AS_config.FLAGS.test_dir3)
		restore_model_file = deep_AS_config.FLAGS.model_dir3
	print("database done")

	with tf.Session() as sess:
		print("Create model for test")
		print("change some para for test")
	#	deep_AS_config.FLAGS.batch_size = 1

		model = deep_AS_model.Model()

		model.saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(restore_model_file)
		if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
			model.saver.restore(sess, ckpt.model_checkpoint_path)
			print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
			print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
		else:
			print("error")
		
		test_cycle(model,sess,x,y,l)
def usage_function():          #使用实际数据，输出预测结果
	print("usage_function")
	if deep_AS_config.FLAGS.model == "use":
		x,l = database_use()
		restore_model_file = deep_AS_config.FLAGS.model_dir
		f = open('pre_label','w')
	elif deep_AS_config.FLAGS.model == "use2":
		x,l = database_use2()
		restore_model_file = deep_AS_config.FLAGS.model_dir2
		f = open('exon','w')
	batch_size = deep_AS_config.FLAGS.batch_size
	with tf.Session() as sess:
		model = deep_AS_model.Model()
		model.saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(restore_model_file)
		if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
			model.saver.restore(sess, ckpt.model_checkpoint_path)
			print("restore model from {0:s}".format(ckpt.model_checkpoint_path))
		else:
			print("error")
		
		number = l//batch_size
		for n in range(number):
			start = batch_size*n
			end = batch_size*n+batch_size
			x_batch=[]
			for i in range(start,end):
				x_batch.append(x[i])
			pre_label = model.usage(sess,x_batch)
			for i in range(len(pre_label)):
				f.write(str(pre_label[i]))
				f.write('\n')
		
def database_use():   #预测剪接位点模型实际数据的处理
	f=open(deep_AS_config.FLAGS.use_dir,'r')
	decodedic = {'A':[0,0,0,1,0],
				'T':[0,0,1,0,0],
				'C':[0,1,0,0,0],
				'G':[1,0,0,0,0],
				'N':[0,0,0,0,1]}
	x_one_hot=[]
	data = f.readlines()
	print("文件的大小有：  ",len(data))
	for line in data:
		arrayx=[]
		x=line.split()[1]
		x = x.upper()
		for i in x:
			arrayx.append(decodedic[i])
		x_one_hot.append(arrayx)
	return x_one_hot,len(x_one_hot)
def database_use2(dir):     #预测外显子模型实际数据的处理
	f=open(deep_AS_config.FLAGS.use_dir2,'r')
	decodedic = {'A':[0,0,0,1,0],
				'T':[0,0,1,0,0],
				'C':[0,1,0,0,0],
				'G':[1,0,0,0,0],
				'N':[0,0,0,0,1]}
	x_one_hot=[]
	data = f.readlines()
	print("训练集的大小有：  ",len(data))
	for line in data:
		#print (len(line.split()))
		arrayx=[]
		x = line.split()[0]+line.split()[1]
		x = x.upper()
		exon_length = int(line.split()[5])
		if 0 < exon_length <=100:
			exon_length = [1,0,0,0,0]
		elif 100 < exon_length <=500:
			exon_length = [0,1,0,0,0]
		elif 500 < exon_length <=1000:
			exon_length = [0,0,1,0,0]
		elif 1000 < exon_length <=2000:
 			exon_length = [0,0,0,1,0]
		elif exon_length > 2000:
 			exon_length = [0,0,0,0,1]
		for i in x:
			if decodedic.__contains__(i):
				a = decodedic[i]
				assert len(a) == 5
				arrayx.append(a)
			else:
				print(i)
				next
		arrayx[-1] = exon_length
		x_one_hot.append(arrayx)
	return x_one_hot,len(x_one_hot)

def LSTMtrain():     #转录本预测模型的训练
	print ("LSTMtrain()")
	dir = "../G_bysj/data/"
	x,y,l=L.database(dir+'protein_transgraph_train',300)
	#x,y,l=L.database(dir+'test',300)
	#print(l)
	epoch = 0
	
	with tf.Session() as sess:
		#件模型
		LSTM = create_RNN_model(sess)
		#记录步数
		global step
		from LSTMclass import step
		#打开log句柄
		log_file = LSTM.log_dir+'log_protein'
		log_file_handle = open(log_file, 'a')
		#保存文件路径
		checkpoint_path = os.path.join(LSTM.model_dir, "protein.ckpt")
		
		losses = []
		while True:
			#规定训练轮数 大于训练轮数便停止
			epoch_last = epoch
			epoch = (step * deep_AS_config.FLAGS.batch_size // l)
			
			train_cycle(LSTM,sess,epoch,checkpoint_path,x,y,l,log_file_handle,losses)

			iftest = epoch-epoch_last
			if iftest:
				
				print("epoch %d : have a test on test-data" %epoch)
				testLSTM(dir+'protein_transgraph_test',LSTM,sess,log_file_handle)
			if epoch >= deep_AS_config.FLAGS.epoch:
				print("EPOCH：%d STOP TRAINING LOOP" %epoch)
				break
		plt.plot(losses)
		plt.savefig("../G_bysj/log/losses.pdf")
		log_file_handle.close()
		#print (lenth)
		#input_feed[self.input_dict["data1_x"]] = input_x
		#input_feed[self.input_dict["data1_y"]] = input_y
		#loss = tf.reduce_mean(self.loss)correct_pred = tf.equal(tf.argmax(self.output, 1),tf.argmax(self.input_dict["data1_y"], 1))
		#accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
def testLSTM(dir,LSTM,sess,log_file_handle):   #转录本预测模型的测试
	print('testLSTM()')
	print('='*80)
	x,y,l = L.database(dir,deep_AS_config.FLAGS.maxlen)
	statistic={}
	statistic['0->1'] = 0 ##把0预测成1 以此类推
	statistic['0->0'] = 0
	statistic['1->0'] = 0
	statistic['1->1'] = 0
	AUC_x = []
	for i in range(l//1000):
		pre = LSTM.teststep(sess,x[i*1000:(i+1)*1000])
		pre = list(pre)
		for i, element in enumerate(pre):
			pre[i] = list(pre[i])
			if y[i].index(1):
				if pre[i].index(max(pre[i])):
					statistic['1->1'] += 1
				else:
					statistic['1->0'] += 1
			else:
				if pre[i].index(max(pre[i])):
					statistic['0->1'] += 1
				else:
					statistic['0->0'] += 1
	auc_x,auc_y = L.getbatch(x,y,l,1000)
	pre = LSTM.teststep(sess,auc_x)
	pre = list(pre)
	for i, element in enumerate(pre):
		pre[i] = list(pre[i])
		AUC_x.append(pre[i][auc_y[i].index(1)])
	str1 = ','.join(str(e) for e in AUC_x)
	print (statistic)
	print (statistic,file=log_file_handle)
	print (str1,file = log_file_handle)
	log_file_handle.flush()
