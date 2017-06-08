import tensorflow as tf
from util.cnn import CNN
from tqdm import tqdm
from util.nn import *

class LabelNet(CNN):
    def build_loss(self):
        #build self.loss and self.opt_op
        #use: self.conv_feats 
        self.clsgt = tf.placeholder(tf.int32,[self.batch_size])
        feashape = self.conv_feats.get_shape().as_list()
        #feadim = feashape[-1]*feashape[-2]*feashape[-3]
        conv_feats_reshape = tf.reshape(self.conv_feats,[feashape[0],-1])
        #self.logits = fully_connected(conv_feats_reshape,46,"cls_logits",init_w='normal', stddev=0.01)
        self.feat_embbed = fully_connected(conv_feats_reshape,4096,"feat_logits",init_w='xavier', stddev=0.01)
        #self.feat_embbed = tf.nn.dropout(self.feat_embbed, 0.5) if self.mode=="train" else self.feat_embbed
        self.logits = fully_connected(self.feat_embbed,46,"cls_logits",init_w='xavier', stddev=0.01)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.clsgt, logits = self.logits)
        self.loss = tf.reduce_mean(self.loss)
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
                img_files, clabels, ctypes = batch
                #feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})
                #feed_dict = self.get_feed_dict_for_all(batch, is_train=True, feats=feats) 
                _, loss, global_step = sess.run([self.opt_op, self.loss, self.global_step], feed_dict={self.img_files:img_files, self.is_train:True, self.clsgt:clabels}) 
                print(" loss=%f " %(loss)) 

                #if (idx+1) % params.test_period == 0:
                #	self.test(sess)
                if (idx+1) % params.save_period == 0:
                    self.save(sess)
                    train_dataset.reset()

        print("Model trained.")
    def eval_embbed(self, sess, imf):
        return sess.run([self.feat_embbed], feed_dict={self.is_train:False, self.img_files:imf})