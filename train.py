import tensorflow as tf
from LabelNet import LabelNet
from TypeNet import TypeNet
#from AttrNet import AttrNet
from AttrNet1000 import AttrNet
from easydict import EasyDict as edict
from util.dataset import DataSet,AttrDataSet

def param_parse(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def train_attrnet():
	params = param_parse("cfgs/AttrNet.yml")
	trainds = AttrDataSet(params.datalist_file,params.path_base,params.batch_size)
	net = AttrNet(params,'train')
	net.set_dataset(trainds,trainds)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()	
		#net.loadfromcaffe("pretrained/vgg16.npy",sess)
		net.load(sess)
		net.train(sess)


def train_labelnet():
	params = param_parse("cfgs/LabelNet.yml")
	net = LabelNet(params,'train')
	trainds = DataSet(params.datalist_file,params.path_base,params.batch_size)
	net.set_dataset(trainds,trainds)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()	
		#net.loadfromcaffe("pretrained/vgg16.npy",sess)
		net.load(sess)
		net.train(sess)

def train_typenet():
	params = param_parse("cfgs/TypeNet.yml")
	net = TypeNet(params,'train')
	trainds = DataSet(params.datalist_file,params.path_base,params.batch_size)
	net.set_dataset(trainds,trainds)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()	
		#net.loadfromnpy("ckpt_npy/typenet.npy",sess)
		#net.loadfromnpy("pretrained/vgg16.npy",sess)
		net.load(sess)
		net.train(sess)

if __name__=='__main__':
	train_typenet()
	#train_labelnet()
	#train_attrnet()
