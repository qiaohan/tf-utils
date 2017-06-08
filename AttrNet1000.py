import tensorflow as tf
from util.cnn import CNN
from tqdm import tqdm
from util.nn import *

class AttrNet(CNN):
    def build_loss(self):
        #build self.loss and self.opt_op
        #use: self.conv_feats 
        self.clsgt = tf.placeholder(tf.float32,[self.batch_size,1000])
        feashape = self.conv_feats.get_shape().as_list()
        #feadim = feashape[-1]*feashape[-2]*feashape[-3]
        conv_feats_reshape = tf.reshape(self.conv_feats,[feashape[0],-1])
        #masks = tf.split(self.masks,self.num_attr,1)
        #clsgt = tf.split(self.clsgt,self.num_attr,1)
        self.feat_embbed = fully_connected(conv_feats_reshape,4096,"feat_logits",init_w='xavier', stddev=0.01)
        self.logits = fully_connected(self.feat_embbed,1000,"cls_logits", init_w='xavier', init_b=0.1, stddev=0.1)
        self.probs = tf.nn.softmax(self.logits)
        labels = self.clsgt
        loss = tf.reduce_sum(labels*tf.log(self.probs)*0.996+(1-labels)*tf.log(1-self.probs)*0.004,reduction_indices=[1])
        self.loss = -tf.reduce_mean(loss)
        self.opt_op = tf.train.AdamOptimizer(self.params.lr).minimize(self.loss,global_step=self.global_step)
    def set_dataset(self,traind,testd):
        self.train_dataset = traind
        self.test_dataset = testd
    def train(self, sess):
        print("Training the model...")
        params = self.params
        train_dataset = self.train_dataset
        for epoch_no in tqdm(list(range(params.num_epochs)), desc='epoch'): 
            for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'): 
                batch = train_dataset.next_batch_for_all() 
                img_files, pos, neg = batch
                #feats = sess.run(self.conv_feats, feed_dict={self.img_files:img_files, self.is_train:False})
                #feed_dict = self.get_feed_dict_for_all(batch, is_train=True, feats=feats) 
                _, loss, global_step = sess.run([self.opt_op, self.loss, self.global_step], feed_dict={self.img_files:img_files, self.is_train:True, self.clsgt:pos}) 
                print(" loss=%f " %(loss)) 

                #if (idx+1) % params.test_period == 0:
                #   self.test(sess)
                if (idx+1) % params.save_period == 0:
                    self.save(sess)
            train_dataset.reset()
        print("Model trained.")