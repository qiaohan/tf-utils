import tensorflow as tf
from LabelNet import LabelNet
from easydict import EasyDict as edict
from util.dataset import DataSet,AttrDataSet

def param_parse(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg
def train_labelnet():
	params = param_parse("cfgs/LabelNet.yml")
	net = LabelNet(params,'train')
	trainds = DataSet(params.datalist_file,params.path_base,params.batch_size)
	net.set_dataset(trainds,trainds)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()	
		#net.loadfromnpy("ckpt_npy/labelnet.npy",sess)
		net.load(sess)
		net.train(sess)
if __name__=='__main__':
	train_labelnet()
