import tensorflow as tf
from LabelNet import LabelNet
from TypeNet import TypeNet
from AttrNet1000 import AttrNet
from easydict import EasyDict as edict
from util.dataset import DataSet

def param_parse(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def test_labelnet():
	params = param_parse("cfgs/LabelNet.yml")
	net = LabelNet(params,'train')
	trainds = DataSet(params.datalist_file,params.path_base,params.batch_size)
	net.set_dataset(trainds,trainds)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()	
		#net.loadfromcaffe("pretrained/vgg16.npy",sess)
		net.load(sess)
		#net.savenpy("ckpt_npy/labelnet.npy",sess)
		net.test(sess)

def test_typenet():
	params = param_parse("cfgs/TypeNet.yml")
	#params = param_parse("cfgs/Test_TypeNet.yml")
	net = TypeNet(params,'test')
	trainds = DataSet(params.datalist_file,params.path_base,params.batch_size)
	net.set_dataset(trainds,trainds)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()	
		#net.loadfromcaffe("pretrained/vgg16.npy",sess)
		net.load(sess)
		net.test(sess)

def test_attrnet():
	params = param_parse("cfgs/AttrNet.yml")
	#params = param_parse("cfgs/Test_TypeNet.yml")
	net = AttrNet(params,'test')
	#trainds = DataSet(params.datalist_file,params.path_base,params.batch_size)
	#net.set_dataset(trainds,trainds)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()	
		#net.loadfromcaffe("pretrained/vgg16.npy",sess)
		net.load(sess)
		net.savenpy("ckpt_npy/attrnet.npy",sess)
		net.test(sess)

if __name__=='__main__':
	#test_typenet()
	#test_labelnet()
	test_attrnet()
