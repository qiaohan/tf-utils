import numpy as np
from random import shuffle
import os

class DataSet(object):
	def __init__(self, lstfile, base, batchsize):
		infos = [l for l in open(lstfile)]
		self.imgs = []
		self.clabels = []
		self.ctypes = [] 
		shuffle(infos)
		for l in infos:
			img,clabel,_,ctype,_ = l.split()
			self.imgs.append(base+img)
			self.clabels.append(int(clabel))
			self.ctypes.append(int(ctype))
		self.pathbase = base
		self.itptr = 0
		self.num_batches = len(self.imgs)/batchsize
		self.batchsize = batchsize
		
	def reset(self):
		self.itptr = 0
	def next_batch_for_all(self):
		#idxs = [ k+self.itptr for k in range(self.batchsize)] 
		imgs = self.imgs[self.itptr:self.itptr+self.batchsize]
		clabels = self.clabels[self.itptr:self.itptr+self.batchsize]
		ctypes = self.ctypes[self.itptr:self.itptr+self.batchsize]
		self.itptr += self.batchsize
		return imgs, clabels, ctypes
class AttrDataSet(object):
	def __init__(self, lstfile, base, batchsize):
		infos = [l for l in open(lstfile)]
		self.imgs = []
		self.masks = []
		self.labels = [] 
		shuffle(infos)
		for l in infos:
			img,mask,label = l.split()
			self.imgs.append(base+img)
			self.labels.append([int(k) for k in label.split(',')])
			self.masks.append([int(k) for k in mask.split(',')])
		#datafile = os.path.join(os.path.split(lstfile)[0],"")
		self.pathbase = base
		self.itptr = 0
		self.num_batches = len(self.imgs)/batchsize
		self.batchsize = batchsize
		
	def reset(self):
		self.itptr = 0
	def next_batch_for_all(self):
		#idxs = [ k+self.itptr for k in range(self.batchsize)] 
		imgs = self.imgs[self.itptr:self.itptr+self.batchsize]
		labels = self.labels[self.itptr:self.itptr+self.batchsize]
		masks = self.masks[self.itptr:self.itptr+self.batchsize]
		self.itptr += self.batchsize
		return imgs, np.asarray(masks), np.asarray(labels)
