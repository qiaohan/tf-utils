'''
import tensorflow as tf
from AttrNet1000 import AttrNet
from easydict import EasyDict as edict
from util.dataset import DataSet
'''
import numpy as np
import pickle,lmdb

def param_parse(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def embbed_attrnet():
	params = param_parse("cfgs/AttrNet.yml")
	params.batch_size=1
	net = AttrNet(params,'test')
	#trainds = DataSet(params.datalist_file,params.path_base,params.batch_size)
	#net.set_dataset(trainds,trainds)
	basepath = "/home/arcthing/qiaohan/data/cloth_data/DeepFashion/CSCRB/Img/"
	output = open('feats_attr.pkl', 'wb')
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		net.load(sess)
		net.savenpy(npyfile = "ckpt_npy/labelnet.npy",sess = sess)
		#net.loadfromcaffe("pretrained/vgg16.npy",sess)
		#feats = {}
		cnt = -3
		for line in open("/home/arcthing/qiaohan/data/cloth_data/DeepFashion/CSCRB/Anno/list_bbox_consumer2shop.txt"):
			cnt+=1
			if cnt<0:
				continue
			line = line.strip().split()[0]
			print basepath+line
			feat = net.eval_embbed(sess, [basepath+line])[0].reshape([-1,])
			#feats[line] = feat
			pickle.dump([line,feat], output)
		#np.save("feats_type",feats)
		output.close()
def convert_pkl2lmdb():
	ff = open('feats_attr.pkl')
	env = lmdb.open("attr_fts", map_size=int(1e12))
	with env.begin(write=True) as txn:
		for i in range(239556):
			key_str,ft = pickle.load(ff)
			if i%5000==0:
				print ft.dtype
			txn.put(key_str, ft.tostring())
if __name__=='__main__':
	#embbed_labelnet()
	embbed_attrnet()
	#convert_pkl2lmdb()
	#embbed_typenet()
	#test_labelnet()
