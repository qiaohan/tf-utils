import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time,os
from nn import *
from easydict import EasyDict as edict

class CNN(object): 
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.batch_size = params.batch_size #if mode=='train' else 1
        self.batch_norm = params.batch_norm
        self.basic_model = params.basic_model

        self.label = self.basic_model + '/'
        self.save_dir = os.path.join(params.save_dir, self.label)

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False) 
        self.conv_feats = None
        self.build() 
        self.loss = None
        self.build_loss()
        self.saver = tf.train.Saver(max_to_keep = 100) 
    def build_loss_opt(self):
        raise NotImplementedError()
    def load_img(self,imgf,isotropic=False):
        file_data = tf.read_file(imgf)
        img = tf.image.decode_jpeg(file_data, channels=self.params.channels)
        #img = tf.reverse(img, [False, False, self.params.bgr]) 
        img_shape = edict()
        new_shape = edict()
        '''
        if isotropic:
            img_shape.h = tf.to_float(tf.shape(img)[0])
            img_shape.w = tf.to_float(tf.shape(img)[1])
            max_length = tf.maximum(img_shape.h, img_shape.w)
            scale_shape = self.params.scale_shape
            new_shape.h = tf.to_int32((scale_shape.h / max_length) * img_shape.h)
            new_shape.w = tf.to_int32((scale_shape.w / max_length) * img_shape.w)
            img = tf.image.resize_image_with_crop_or_pad(img, new_shape.h, new_shape.w)
        else:
            new_shape = self.params.scale_shape
            img = tf.image.resize_image(img, [new_shape.h, new_shape.w])
        '''

        #crop_shape = self.params.crop_shape
        #offset = [(new_shape.h - crop_shape.h) / 2, (new_shape.w - crop_shape.w) / 2]
        #img = tf.slice(img, tf.to_int32([offset[0], offset[1], 0]), tf.to_int32([crop_shape.h, crop_shape.w, -1]))

        new_shape = self.params.scale_shape
        img = tf.image.resize_image_with_crop_or_pad(img, new_shape.h, new_shape.w)
        #img = tf.image.resize_images(img, [new_shape.h, new_shape.w])
        img = tf.to_float(img)-[120,120,120]
        return img  
    def build(self):
        if self.basic_model=='vgg16':
            self.build_basic_vgg16()

        elif self.basic_model=='resnet50':
            self.build_basic_resnet50()

        elif self.basic_model=='resnet101':
            self.build_basic_resnet101()

        else:
            self.build_basic_resnet152()

    def build_basic_vgg16(self):
        print("Building the basic VGG16 net...")
        bn = self.batch_norm
        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)
        print "batch size:", self.batch_size
        imgs = []
        for img_file in tf.split(img_files,[1,]*self.batch_size):
            img_file = tf.squeeze(img_file)
            imgs.append(self.load_img(img_file,True))
        imgs = tf.stack(imgs) 

        print "images batch shape:",imgs.get_shape().as_list()

        convole = convolution_no_bias if bn else convolution
        conv1_1_feats = convole(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
        conv1_2_feats = convole(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = convole(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
        conv2_2_feats = convole(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
        pool2_feats = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = convole(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
        conv3_2_feats = convole(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
        conv3_3_feats = convole(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
        pool3_feats = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = convole(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
        conv4_2_feats = convole(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
        conv4_3_feats = convole(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
        pool4_feats = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = convole(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
        conv5_2_feats = convole(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
        conv5_3_feats = convole(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

        self.conv_feats = conv5_3_feats

        self.img_files = img_files
        self.is_train = is_train
        print("Basic VGG16 net built.")

    def basic_block(self, input_feats, name1, name2, is_train, bn, c, s=2):
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, bn, None)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, bn, c):
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_basic_resnet50(self):
        print("Building the basic ResNet50 net...")
        bn = self.batch_norm

        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)

        imgs = []
        for img_file in tf.unpack(img_files):
            imgs.append(load_img(img_file))
        imgs = tf.pack(imgs)          

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)
        res3b_feats = self.basic_block2(res3a_feats, 'res3b', 'bn3b', is_train, bn, 128)
        res3c_feats = self.basic_block2(res3b_feats, 'res3c', 'bn3c', is_train, bn, 128)
        res3d_feats = self.basic_block2(res3c_feats, 'res3d', 'bn3d', is_train, bn, 128)

        res4a_feats = self.basic_block(res3d_feats, 'res4a', 'bn4a', is_train, bn, 256)
        res4b_feats = self.basic_block2(res4a_feats, 'res4b', 'bn4b', is_train, bn, 256)
        res4c_feats = self.basic_block2(res4b_feats, 'res4c', 'bn4c', is_train, bn, 256)
        res4d_feats = self.basic_block2(res4c_feats, 'res4d', 'bn4d', is_train, bn, 256)
        res4e_feats = self.basic_block2(res4d_feats, 'res4e', 'bn4e', is_train, bn, 256)
        res4f_feats = self.basic_block2(res4e_feats, 'res4f', 'bn4f', is_train, bn, 256)

        res5a_feats = self.basic_block(res4f_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.img_files = img_files
        self.is_train = is_train
        print("Basic ResNet50 net built.")

    def build_basic_resnet101(self):
        print("Building the basic ResNet101 net...")
        bn = self.batch_norm

        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)

        imgs = []
        for img_file in tf.unpack(img_files):
            imgs.append(load_img(img_file))
        imgs = tf.pack(imgs)  

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 4):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b3_feats = temp
 
        res4a_feats = self.basic_block(res3b3_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 23):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b22_feats = temp

        res5a_feats = self.basic_block(res4b22_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.img_files = img_files
        self.is_train = is_train
        print("Basic ResNet101 net built.")

    def build_basic_resnet152(self):
        print("Building the basic ResNet152 net...")
        bn = self.batch_norm

        img_files = tf.placeholder(tf.string, [self.batch_size])
        is_train = tf.placeholder(tf.bool)

        imgs = []
        for img_file in tf.unpack(img_files):
            imgs.append(load_img(img_file))
        imgs = tf.pack(imgs)  

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 8):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b7_feats = temp
 
        res4a_feats = self.basic_block(res3b7_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 36):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b35_feats = temp

        res5a_feats = self.basic_block(res4b35_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.img_files = img_files
        self.is_train = is_train
        print("Basic ResNet152 net built.")

    def load(self, sess):
        print("Loading model...") 
        checkpoint = tf.train.get_checkpoint_state(self.save_dir) 
        if checkpoint is None: 
            print("Error: No saved model found. Please train first.") 
            sys.exit(0) 
        self.saver.restore(sess, checkpoint.model_checkpoint_path) 
    def loadfromnpy(self, data_path, session, ignore_missing=True):
        print("Loading basic model from %s..." %data_path)
        for v in tf.trainable_variables():
            print v.name
        data_dict = np.load(data_path).item()
        count = 0
        miss_count = 0
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                        print("Variable %s:%s loaded" %(op_name, param_name))
                    except ValueError,e:
                        miss_count += 1
                        print("Variable %s:%s missed" %(op_name, param_name))
                        print(e)
            if not ignore_missing:
                raise
        print("%d variables loaded. %d variables missed." %(count, miss_count))
    def save(self, sess, npyfile=None):
        print("Saving model to %s" %self.save_dir)
        self.saver.save(sess, self.save_dir, 10) #self.global_step)
        if npyfile is not None:
            self.savenpy(npyfile, sess)
    def savenpy(self, npyfile, sess):
        print("saving the model params to .npy file:"+npyfile)
        data_dict = {}
        op_name = []
        for v in tf.trainable_variables():
            opname, vname = v.name.split('/')[-2:]
            vname = vname.split(':')[0]
            vdata = v.eval(session = sess)
            print opname,vname,vdata.shape
            if opname in op_name:
                data_dict[opname][vname] = vdata.copy()
            else:
                data_dict[opname] = {vname:vdata.copy()}
                op_name.append(opname)
        np.save(npyfile,data_dict)
    def eval_feat(self, sess, imf):
        return sess.run([self.conv_feats], feed_dict={self.is_train:False, self.img_files:imf})
    def eval_embbed(self, sess, imf):
        return sess.run([self.feat_embbed], feed_dict={self.is_train:False, self.img_files:imf})
