import tensorflow as tf
from util.cnn import CNN
from tqdm import tqdm
from util.nn import *

class AttrNet(CNN):
	def build_loss(self):
		#build self.loss and self.opt_op
		#use: self.conv_feats 
		self.num_attr = 1000
		self.clsgt = tf.placeholder(tf.int32,[self.batch_size,self.num_attr])
		self.masks = tf.placeholder(tf.float32,[self.batch_size,self.num_attr])
		feashape = self.conv_feats.get_shape().as_list()
    		#feadim = feashape[-1]*feashape[-2]*feashape[-3]
    		conv_feats_reshape = tf.reshape(self.conv_feats,[feashape[0],-1])
		masks = [tf.squeeze(k) for k in tf.split(self.masks,self.num_attr,1)]
		clsgt = [tf.one_hot(tf.squeeze(k),2) for k in tf.split(self.clsgt,self.num_attr,1)]
		#masks = tf.split(self.masks,self.num_attr,1)
		#clsgt = tf.split(self.clsgt,self.num_attr,1)
		self.loss = []
		for i in range(self.num_attr):
			print "building attr:",i	
			logits = fully_connected(conv_feats_reshape,2,"cls_logits_"+str(i), init_w='xavier', stddev=0.1, init_b=0.1)
			print "label shape:",clsgt[i].get_shape().as_list()				
			p = tf.nn.softmax(logits)	
			loss = -tf.reduce_sum(clsgt[i] * tf.log(p)*0.996 + (1-clsgt[i])*tf.log(1-p)*0.004 ,reduction_indices=[1]) * masks[i]
			self.loss.append(tf.reduce_mean(loss))
		self.loss = tf.reduce_sum(tf.stack(self.loss))
		self.opt_op = tf.train.AdamOptimizer(self.params.lr).minimize(self.loss,global_step=self.global_step)
		#correct_prediction = tf.equal(tf.argmax(logits,1), self.clsgt)
		#self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	def set_dataset(self,traind,testd):
		self.train_dataset = traind
		self.test_dataset = testd
	def test(self, sess):
                correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits,1),tf.int32), self.clsgt)
                self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                test_dataset = self.test_dataset
                total_acc = 0
                for idx in tqdm(list(range(test_dataset.num_batches)), desc='test'): 
                        batch = test_dataset.next_batch_for_all() 
                        img_files, clabels, ctypes = batch
                        [acc] = sess.run([self.acc], feed_dict={self.img_files:img_files, self.is_train:False, self.clsgt:clabels}) 
                        print("accuracy=%f " %(acc)) 
                        total_acc += acc
                total_acc /= test_dataset.num_batches
                print("total accuracy: %f" %(total_acc))
                test_dataset.reset()
	def train(self, sess):
        	print("Training the model...")
        	params = self.params
		train_dataset = self.train_dataset
        	for epoch_no in tqdm(list(range(params.num_epochs)), desc='epoch'): 
            		for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'): 
                		batch = train_dataset.next_batch_for_all() 
                		img_files, masks, labels = batch
                		#feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})
                		#feed_dict = self.get_feed_dict_for_all(batch, is_train=True, feats=feats) 
                		_, loss, global_step = sess.run([self.opt_op, self.loss, self.global_step], feed_dict={self.img_files:img_files, self.is_train:True, self.masks:masks, self.clsgt:labels}) 
                		print(" loss=%f " %(loss)) 

            			#if (idx+1) % params.test_period == 0:
            			#	self.test(sess)
				if (idx+1) % params.save_period == 0:
                			self.save(sess)
            		train_dataset.reset()

        	print("Model trained.")
