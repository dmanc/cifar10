import os
import random
import numpy as np
import skimage.transform, skimage.io, skimage.color
import matplotlib.pyplot as plt
import pandas as pd

_image_extension = ("jpg", "jpeg", "png")
class image_db():
	### load images from directory imgdir
	""" kwargs:
		- train_portion			portion of dataset for training (max 1.0)
		- seed					random seed for reproducibility
	"""
	def __init__(self, imgdir, *args, **kwargs):
		### get kwargs ###
		train_portion = kwargs.get('train_portion', 0.5)
		seed = kwargs.get('seed', None)

		### set env ###
		np.random.seed(seed)

		### discover images ###
		self.imgdir = imgdir
		self.fnames = []
		for path in os.listdir(self.imgdir):
			if not os.path.isdir(path) and path.endswith(_image_extension):
				self.fnames.append(os.path.join(self.imgdir, path))
		self.images = skimage.io.imread_collection(self.fnames, conserve_memory=True)
		self.size = len(self.images)
		print self.size, "images loaded from", self.imgdir

		### init pools for each mode ###
		self.modeval = ['train', 'test', 'all', 'list']
		self.is_shuffle = [True, True, True, False]
		self.imgindex_cnt = [0] * len(self.modeval)
		self.imgindex = [list(range(self.size)) for _ in range(len(self.modeval))]
		for m, iindex in enumerate(self.imgindex):
			if self.is_shuffle[m]: random.shuffle(iindex)

		### split train/test ###
		index_split = int(self.size*train_portion)
		self.imgindex[1] = self.imgindex[0][index_split:]
		self.imgindex[0] = self.imgindex[0][:index_split]

	def transform_label(self, labels_map):
		self.tlabel_map = labels_map

	def get_index(self, idx, size=None, cmap='rgb'):
		### get images ###
		img = self.images[idx]

		### convert to desired cmap ###
		img = self._rgb2cmap(img, cmap)

		### resize if necessary
		if size is not None: img = self._img_resize(img, size)

		return img, self.tlabel_map(os.path.splitext(os.path.basename(self.fnames[idx]))[0])

	def get_batch(self, num, mode='all', size=None, cmap='rgb'):
		if mode not in self.modeval:
			raise ValueError(mode + " not in " + str(self.modeval))
		m = self.modeval.index(mode)
		batch_imgs, batch_labels = [], []
		for i in range(num):
			### get image and modify ###
			img, label = self.get_index(self.imgindex[m][self.imgindex_cnt[m]], size, cmap)
			batch_imgs.append(img)
			batch_labels.append(label)

			### update indexing ###
			self.imgindex_cnt[m] += 1
			if self.imgindex_cnt[m] == len(self.imgindex[m]):
				if self.is_shuffle[m]: random.shuffle(self.imgindex[m])
				self.imgindex_cnt[m] = 0
		batch_imgs = np.array(batch_imgs)
		batch_labels = np.array(batch_labels)
		return batch_imgs, batch_labels

	def get_size(self):
		return self.size

	def _rgb2cmap(self, img, cmap):
		if cmap == 'grey':
			img = skimage.color.rgb2grey(img)
			img = img.reshape((img.shape[0], img.shape[1], 1))
		elif cmap == 'hsv':
			img = skimage.color.rgb2hsv(img)
		return img

	def _img_resize(self, img, size):
		resized_img = np.zeros(size + [img.shape[2]])
		for k in range(img.shape[2]):
			resized_img[..., k] = skimage.transform.resize(img[..., k], size, mode='reflect')
		return resized_img

class Label():
	def __init__(self, id2num, num2name, name2num):
		self.id2num = id2num
		self.num2name = num2name
		self.name2num = name2num

	def i2s(self, i):
		return self.num2name[self.id2num[i]]

	def i2n(self, i):
		return self.id2num[i]

	def n2s(self, i):
		return self.num2name[i]

	def s2n(self, i):
		return self.name2num[i]

def cifar10_read_label(csv_fname):
	df = pd.read_csv(csv_fname)

	### explore labels ###
	unique_label = df.label.unique()
	num2name = {i:l for i, l in zip(range(len(unique_label)), unique_label)}
	name2num = {l:i for i, l in zip(range(len(unique_label)), unique_label)}
	print 'num2name', num2name
	print 'name2num', name2num

	### convert to num label ###
	df = df.replace({'label':name2num})
	id2num = pd.Series(df.label.values,index=df.id.astype(str)).to_dict()

	return Label(id2num, num2name, name2num)

if __name__ == '__main__':
	labels = cifar10_read_label("../trainLabels.csv")

	idb = image_db("../train")
	idb.transform_label(lambda x: labels.i2s(x))
	size = [100, 100]
	mode = 'test'
	cmap = 'grey'
	batch_img, batch_label = idb.get_batch(16, size=size, mode=mode, cmap=cmap)
	print batch_label
	bwimages = [batch_img[i][:, :, 0].astype(float)/255.0 for i in range(16)]
	plt.figure(figsize=(10, 10))
	for i in range(16):
	    plt.subplot(4, 4, i+1)
	    if cmap == 'grey':
	    	plt.imshow(bwimages[i], cmap='gray')
	    else:
	    	plt.imshow(batch_img[i])
	    plt.title(batch_label[i])
	plt.show()
	plt.close()