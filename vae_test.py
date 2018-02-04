import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
import pandas as pd

from vae_build import Vae
from dataset import image_db, cifar10_read_label
from sklearn.decomposition import PCA

BATCH_SIZE = 100
def plot_all_2d(sess, vae, idb):
	### get images in batch ###
	idb_size = idb.get_size()
	all_latents, all_labels = [], []
	for i in range(0, idb_size, BATCH_SIZE):
	# for i in range(0, 1000, BATCH_SIZE):
		### get a batch ###
		batch_size = BATCH_SIZE if i+BATCH_SIZE <= idb_size else idb_size-i
		batch_img, batch_label = idb.get_batch(batch_size, size=vae.in_size, \
			mode='list', cmap='grey')
		batch_label = [int(x.split('.')[0][5:]) for x in batch_label]

		### transform to latent vars ###
		latents = vae.transform(sess, batch_img)
		all_latents.extend(latents)
		all_labels.extend(batch_label)

		print "batch", i
	all_latents = np.array(all_latents)
	all_labels = np.array(all_labels)
	# print all_labels, all_latents

	### get in groups ###
	latents_groups = []
	for lb in list(set(all_labels)):
		latents_groups.append(all_latents[all_labels == lb])
	latents_groups = np.array(latents_groups)

	### plot by groups ###
	for lg, lb in zip(latents_groups, list(set(all_labels))):
		plt.scatter(lg[:, 0], lg[:, 2], label='style ' + str(lb), alpha=0.3)
	leg = plt.legend(ncol=2, prop={'size': 15})
	for lh in leg.legendHandles:
		lh.set_alpha(1.0)
	plt.figure()
	for lg, lb in zip(latents_groups, list(set(all_labels))):
		plt.scatter(lg[:, 0], lg[:, 2], label='style ' + str(lb), alpha=0.1)
	leg = plt.legend(ncol=2, prop={'size': 15})
	for lh in leg.legendHandles:
		lh.set_alpha(1.0)

	# ### build df ###
	# latent_names = ["l_"+str(i) for i in range(vae.nlatent)]
	# df = pd.DataFrame(all_latents, columns=latent_names)
	# df["style"] = pd.Series(all_labels, index=df.index)
	# # print df

	# g = sns.pairplot(df, vars=latent_names, hue="style", diag_kind='kde',
	# 				plot_kws=dict(s=10, linewidth=0, alpha=0.5))
	# g.set(xticklabels=[], yticklabels=[])

	plt.show()

NX, NY = 15, 15
P = 2
def plot_2dpca(sess, vae, idb):

	### plot 2d tiles ###
	xs = np.linspace(-3, 3, NX)
	ys = np.linspace(-3, 3, NY)
	size_x = vae.in_size[0]+2*P
	size_y = vae.in_size[1]+2*P
	canvas = np.zeros((size_x*NX, size_y*NY))
	for i, xi in enumerate(xs):
		for j, yj in enumerate(ys):
			latents = np.array([[xi, yj] + [0.0]*(vae.nlatent - 2)])
			img = vae.generate(sess, latents)
			canvas[i*size_x+P:(i+1)*size_x-P, j*size_y+P:(j+1)*size_y-P] = \
				img[0].reshape(size_x-2*P, size_y-2*P)
	plt.figure(figsize=(12, 9)); plt.imshow(canvas, cmap='gray'); plt.axis('off')

	plt.show()

if __name__ == '__main__':
	# dataset = image_db('wrenches')
	# vae = Vae(name="2d_vae_5", dataset=dataset, batch_size=10, epoch_num=300, \
	# 	in_size=[128, 128], \
	# 	cnvlf=[64, 32, 16], kernel_size=[[10,10], [5,5], [5,5]], strides=[2, 2, 2], \
	# 	nlatent=8, lr=0.001)
	
	# dataset = image_db('springs/all')
	# vae = Vae(name="2d_vae_sps_1", dataset=dataset, batch_size=50, epoch_num=300, \
	# 	in_size=[150, 150], \
	# 	cnvlf=[64, 64], kernel_size=[[10,10], [5,5]], strides=[5, 2], \
	# 	d_cnvlf=[64, 1], d_kernel_size=[[5,5], [10,10]], d_strides=[2, 5], \
	# 	nlatent=16, lr=0.001)
	# vae = Vae(name="2d_vae_sps_2", dataset=dataset, batch_size=50, epoch_num=300, \
	# 	in_size=[150, 150], \
	# 	cnvlf=[16, 32, 64, 128], kernel_size=[[10,10], [10,10], [5,5], [5,5]], strides=[1, 5, 3, 2], \
	# 	d_cnvlf=[64, 32, 16, 1], d_kernel_size=[[5,5], [5,5], [10,10], [10,10]], d_strides=[2, 3, 5, 1], \
	# 	nlatent=32, lr=0.001)

	labels = cifar10_read_label("../trainLabels.csv")
	dataset = image_db('../train', train_portion=0.8, seed=42)
	dataset.transform_label(lambda x: labels.i2n(x))
	vae = Vae(name="2d_vae_0", in_size=[32, 32], \
		cnvlf=[8, 16], kernel_size=[[5,5], [5,5]], strides=[2, 2], \
		d_cnvlf=[8, 1], d_kernel_size=[[5,5], [5,5]], d_strides=[2, 2], \
		nlatent=8, lr=0.001)

	sess = tf.InteractiveSession()
	vae.restore(sess)

	# plot_slider(sess, vae, dataset, -3.0, 3.0, 4, 3, ispca=False)
	plot_all_2d(sess, vae, dataset)
	# plot_2dpca(sess, vae, dataset)