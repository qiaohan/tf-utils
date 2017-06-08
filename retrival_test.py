import pickle
import heapq,os
from easydict import EasyDict as edict
import numpy as np
class maxHeap:
    def __init__(self, topk, key=lambda x: x):
        self.k = topk
        self.key = key
        self._data =[]

    def push(self, item):
        if len(self._data) <= self.k:
            heapq.heappush(self._data, (self.key(item), item))
        else:
            topk_small = list(self._data[0])
            if item.dist > topk_small[1].dist:
                heapq.heapreplace(self._data, (self.key(item), item))

    def pop(self):
        if len(self._data) > 1:
            return heapq.heappop(self._data)[1]
        else:
            return None

    def topk(self):
        return [x for x in [self.pop() for x in xrange(len(self._data)-1)]]

ff = open('feats_attr.pkl')
numfiles = 239556

inp = 300
inp_feat = []
inp_name = []
for i in range(numfiles):
	feats = pickle.load(ff)
	if i==inp:
		inp_name,inp_feat = feats
		break
print inp_feat
#retrieval
ff.close()
ff = open('feats_attr.pkl')
#ff = open('feats_clabel.pkl')
heap = maxHeap(20, key=lambda x: x.dist)
items = []
for i in range(numfiles):
	feats = pickle.load(ff)
	item = edict()
	#print feats
	item.dist = np.sum( np.square(feats[1]-inp_feat) )
	item.name = feats[0]
	if item.dist == 0:
		print item
	if i%5000==0:
		print i
	items.append(item)
	#heap.push(item)
'''
print inp_name
topk = heap.topk()
topk = [k.name for k in topk]
print topk
'''
topk = sorted(items, key=lambda y: y.dist)
topk = topk[:20]
topk = [k.name for k in topk]

basepath = "/home/arcthing/qiaohan/data/cloth_data/DeepFashion/CSCRB/Img/"
resdir = "retrival_res"+str(inp)+'/'
os.mkdir(resdir)
os.symlink(basepath+inp_name,resdir+'inp_'+'_'.join(inp_name.split('/')))
i=0
for fname in topk:
	i+=1
	os.symlink(basepath+fname,resdir+'res_'+str(i)+'_'.join(fname.split('/')))
