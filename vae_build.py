import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

DIR_SMODEL = 'saved_model/2d_vae'

import matplotlib.pyplot as plt
from skimage import transform
from sklearn.decomposition import PCA
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from dataset import image_db, cifar10_read_label

def convolution2d(inputs, filters, kernel_size, strides, name, activation=tf.nn.relu):
	return tf.layers.conv2d(inputs=inputs, \
		filters=filters, \
		kernel_size=kernel_size, \
		strides=strides, \
		padding='SAME', \
		kernel_initializer=tf.contrib.layers.xavier_initializer(), \
		bias_initializer=tf.zeros_initializer(), \
		# bias_initializer=tf.constant_initializer(0.1, dtype=tf.float32), \
		activation=tf.nn.relu, \
		name=name)

def convolution2d_T(inputs, filters, kernel_size, strides, name, activation=tf.nn.relu):
	return tf.layers.conv2d_transpose(inputs=inputs, \
		filters=filters, \
		kernel_size=kernel_size, \
		strides=strides, \
		padding='SAME', \
		kernel_initializer=tf.contrib.layers.xavier_initializer(), \
		bias_initializer=tf.zeros_initializer(), \
		# bias_initializer=tf.constant_initializer(0.1, dtype=tf.float32), \
		activation=activation, \
		name=name)

### inputs -> 32 x 32 x 1 ###
def encoder(inputs, *args, **kwargs):
	### load setting ###
	filters = kwargs.get("filters", [32, 16])
	kernel_size = kwargs.get("kernel_size", [[5,5], [5,5]])
	strides = kwargs.get("strides", [2, 2])
	nlayer = len(filters)
	if len(kernel_size) != nlayer and len(strides) != nlayer:
		raise VauleError("argument nlayer not match")

	### encoder -> 16x16x32 -> 8x8x16 ###
	with tf.name_scope('Encoder') as scope:
		prev_layer = inputs
		enc_all = []
		for i in range(nlayer):
			f = filters[i]
			ks = kernel_size[i]
			st = strides[i]
			prev_layer = convolution2d(prev_layer, f, ks, strides=st, name=scope+"conv"+str(i+1))
			enc_all.append(prev_layer)
		enc_summary = []
		for ie, enc in enumerate(enc_all):
			size1, size2 = int(enc.shape[1]), int(enc.shape[2])
			for i in range(enc.shape[3]):
				enc_summary.append( tf.summary.image('enc_'+str(ie+1)+'_'+str(i), \
					tf.reshape(enc[..., i], [1, size1, size2, 1]), max_outputs=1))
	return enc_all[-1], {'image': enc_summary, 'layer':enc_all}

### inputs -> 2 x 2 x 8 ###
def decoder(inputs, *args, **kwargs):
	### load setting ###
	filters = kwargs.get("filters", [32, 1])
	kernel_size = kwargs.get("kernel_size", [[5,5], [5,5]])
	strides = kwargs.get("strides", [2, 2])
	nlayer = len(filters)
	if len(kernel_size) != nlayer and len(strides) != nlayer:
		raise VauleError("argument nlayer not match")

	### decoder -> 16x16x32 -> 32x32x1 ###
	with tf.name_scope('Decoder') as scope:
		prev_layer = inputs
		dec_all = []
		for i in range(nlayer):
			f = filters[i]
			ks = kernel_size[i]
			st = strides[i]
			act_fn = tf.nn.relu if i < nlayer-1 else tf.nn.sigmoid
			prev_layer = convolution2d_T(prev_layer, f, ks, strides=st, \
				name=scope+"deconv"+str(nlayer-i), activation=act_fn)
			dec_all.append(prev_layer)
		dec_summary = []
		for ie, dec in enumerate(dec_all):
			size1, size2 = int(dec.shape[1]), int(dec.shape[2])
			for i in range(dec.shape[3]):
				dec_summary.append( tf.summary.image('dec_'+str(len(dec_all)-ie)+'_'+str(i), \
					tf.reshape(dec[..., i], [1, size1, size2, 1]), max_outputs=1))
	return dec_all[-1], {'image': dec_summary, 'layer':dec_all}

### inputs -> 8 x 8 x 16 ###
def recognizer(inputs, outdim=32):
	indim = int(inputs.shape[1]*inputs.shape[2]*inputs.shape[3])
	### recognizer -> 32 (2x2x8) ###
	with tf.name_scope('Recognizer') as scope:
		### prepare ###
		inputs_flat = tf.reshape(inputs, [-1, indim])

		### mean ###
		W_mean = tf.get_variable(scope+"W_mean", shape=[inputs_flat.shape[1], outdim], \
			initializer=tf.contrib.layers.xavier_initializer())
		b_mean = tf.Variable(tf.constant(0.1, shape=[outdim]), name="b_mean")
		z_mean = tf.matmul(inputs_flat, W_mean) + b_mean

		### log sigma squared ###
		W_log_sigma_sq = tf.get_variable(scope+"W_lss", shape=[inputs_flat.shape[1], outdim], \
			initializer=tf.contrib.layers.xavier_initializer())
		b_log_sigma_sq = tf.Variable(tf.constant(0.1, shape=[outdim]), name="b_lss")
		z_log_sigma_sq = tf.matmul(inputs_flat, W_log_sigma_sq) + b_log_sigma_sq
	return z_mean, z_log_sigma_sq

### inputs -> 32 (2x2x8) ###
def generator(inputs, outshape=[8, 8, 16]):
	outdim = outshape[0]*outshape[1]*outshape[2]
	### generator -> 32 (2x2x8) ###
	with tf.name_scope('Generator') as scope:
		### map back ###
		W_1 = tf.get_variable(scope+"W_1", shape=[inputs.shape[1], outdim], \
			initializer=tf.contrib.layers.xavier_initializer())
		b_1 = tf.Variable(tf.constant(0.1, shape=[outdim]), name="b_1")
		y_1 = tf.matmul(inputs, W_1) + b_1

		### reshape to image(s) ###
		y_reshaped = tf.reshape(y_1, [-1] + outshape)
	return y_reshaped

### draw sample from a distribution ###
def distrib_sampler(z_mean, z_lss):
	with tf.name_scope('DSampler') as scope:
		eps = tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
		z = z_mean + tf.sqrt(tf.exp(z_lss)) * eps # z = mu + sigma*eps
	return z

def listify(cllt, t):
	return [t(it) for it in cllt]

class Vae():

	### initialization ###
	"""
	Kwargs
		name:			scope name, name of this structure
		dataset:		source of datase
		batch_size:		number of images per batch
		epoch_num:		number of iteration over all images in dataset
		in_size:		input image size
		cnvlf:			convolution filters for each layer
		kernel_size:	kernel window size for each layer
		strides:		skipping distance for each layer
		nlatent:		dimension fo latent variable
		d_cnvlf:		convolution filters for each layer (for conv_T)
		d_kernel_size:	kernel window size for each layer (for conv_T)
		d_strides:		skipping distance for each layer (for conv_T)
		lr:				learning rate
	"""
	def __init__(self, *args, **kwargs):
		### setup checkpoint env ###
		self.name_scope = kwargs.get('name', "2d_vae")
		self.save_path = os.path.join(DIR_SMODEL, self.name_scope)
		if not os.path.exists(self.save_path):
			self.report("creating " + self.save_path)
			os.makedirs(self.save_path)

		### metadata ###
		self.in_size = kwargs.get('in_size', [32, 32])
		self.pca = None
		self.is_trained = False

		### load setting ###
		self.cnvlf = kwargs.get('cnvlf', [32, 16])
		self.kernel_size = kwargs.get('kernel_size', [[5,5], [5,5]])
		self.strides = kwargs.get('strides', [2, 2])
		self.nlatent = kwargs.get('nlatent', 32)
		self.d_cnvlf = kwargs.get('d_cnvlf', [32, 1])
		self.d_kernel_size = kwargs.get('d_kernel_size', [[5,5], [5,5]])
		self.d_strides = kwargs.get('d_strides', [2, 2])
		self.lr = kwargs.get('lr', 0.001)

		### build autoencoder ###
		with tf.name_scope(self.name_scope):
			### autoencoder stack structure ###
			self.vae_inputs = tf.placeholder(tf.float32, [None] + self.in_size + [1], name='dataset_img')

			### 2d convolutional encoder ###
			self.vae_latents, self.kwret_enc = encoder(self.vae_inputs, filters=self.cnvlf, \
				kernel_size=self.kernel_size, strides=self.strides)

			### distribution explorer ###
			self.recog_mean, self.recog_lss = recognizer(self.vae_latents, self.nlatent)

			### distribution sampler ###
			self.z = distrib_sampler(self.recog_mean, self.recog_lss)

			### distribution interpretator ###
			self.gen_latents = generator(self.z, outshape=listify(self.vae_latents.shape[1:], int))

			### 2d de-convolution image recovery ###
			self.vae_outputs, self.kwret_dec = decoder(self.gen_latents, \
				filters=self.d_cnvlf, kernel_size=self.d_kernel_size, \
				strides=self.d_strides)

			### training spec ###
			with tf.name_scope('loss_function'):
				### negative log prob: Bernoulli ###
				self.clipped_output = tf.clip_by_value(self.vae_outputs, 1e-7, 1-1e-7)
				self.reconstr_loss = -tf.reduce_sum( \
					self.vae_inputs * tf.log(self.clipped_output) \
					+ (1-self.vae_inputs) * tf.log(1.0 - self.clipped_output))

				### rmse ###
				# self.reconstr_loss = tf.reduce_mean(tf.square(self.vae_inputs - self.vae_outputs))

				### kl-divergence ###
				self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.recog_mean) \
					+ tf.exp(self.recog_lss) - self.recog_lss - 1)

				### all loss sum ###
				self.loss = self.reconstr_loss + self.latent_loss
			self.train_ae = tf.train.AdamOptimizer(self.lr).minimize(self.loss)	

			### init ###
			self.init = tf.global_variables_initializer()

	### get saver ###
	def get_saver(self):
		var_list = [var for var in tf.global_variables() \
			if var.name.startswith(self.name_scope)]
		saver = tf.train.Saver(var_list)
		ckpt = tf.train.get_checkpoint_state(self.save_path)
		return saver, ckpt

	### restore trained model ###
	def restore(self, sess):
		self.report("starting restore")
		saver, ckpt = self.get_saver()
		if ckpt and ckpt.model_checkpoint_path:
			self.report("found saved checkpoint in " + str(ckpt.model_checkpoint_path))
			saver.restore(sess, ckpt.model_checkpoint_path)
			self.is_trained = True
		else:
			self.report("no checkpoint found")
		self.report("done")

	### train model ###
	def train(self, sess, dataset, *args, **kwargs):
		self.report("starting train")

		### load database ###
		batch_size = kwargs.get('batch_size', 50)
		epoch_num = kwargs.get('epoch_num', 100)
		if dataset:
			nbatch_per_ep_train = int(dataset.get_size('train') / batch_size)
			nbatch_per_ep_test = int(dataset.get_size('test') / batch_size)

		### grab saver ###
		saver, ckpt = self.get_saver()

		### training ###
		for ep in range(epoch_num):
			sm_loss = [0.0] * 3
			for bn in range(nbatch_per_ep_train):
				batch_img, batch_label = dataset.get_batch(batch_size, \
					mode='train', size=self.in_size, cmap='grey')
				loss_out = sess.run([self.train_ae, self.loss, self.reconstr_loss, self.latent_loss], \
					feed_dict={self.vae_inputs:batch_img})
				loss_out[1:] = [t/batch_size for t in loss_out[1:]]
				sm_loss = [s+t for s, t in zip(sm_loss, loss_out[1:])]
				print "ep=", ep, "bn=", bn, ":", loss_out[1:]
			print ">>> average: bn=", bn, ":", [s/nbatch_per_ep_train for s in sm_loss]

			sm_loss = [0.0] * 3
			for bn in range(nbatch_per_ep_test):
				batch_img, batch_label = dataset.get_batch(batch_size, \
					mode='test', size=self.in_size, cmap='grey')
				loss_out = sess.run([self.loss, self.reconstr_loss, self.latent_loss], \
					feed_dict={self.vae_inputs:batch_img})
				loss_out = [t/batch_size for t in loss_out]
				sm_loss = [s+t for s, t in zip(sm_loss, loss_out)]
				print "TEST; ep=", ep, "bn=", bn, ":", loss_out
			print ">>> TEST average: bn=", bn, ":", [s/nbatch_per_ep_test for s in sm_loss]

			save_f = saver.save(sess, os.path.join(self.save_path, 'model.ckpt'))
			self.report("saved model:" + str(save_f))
		self.report("training done")

	### write summary: graph + layers sample + latent embedding ###
	def write_summary(self, sess, filename="viz/autoencoder/temp"):
		self.report("starting write_summary")

		### create tf writer + write graph ###		
		writer = tf.summary.FileWriter(filename, graph=sess.graph)
		self.report("graph written")

		### plot layers of a run ###
		# self.report("start drawing layers")
		# batch_img, batch_label = dataset.get_batch(1, mode='test', size=self.in_size)
		# recon_img = sess.run([self.vae_outputs], feed_dict={self.vae_inputs:batch_img})[0]
		# for img_sum in self.kwret_enc['image']+self.kwret_dec['image']:
		#	writer.add_summary(img_sum.eval(feed_dict={self.vae_inputs:batch_img}))
		# self.report("layers drawn")

		writer.flush()
		self.report("all summary saved")

	def report(self, msg):
		print  "[{}]: {}".format(self.name_scope, msg)

	def transform(self, sess, X):
		z_mu = sess.run(self.recog_mean, feed_dict={self.vae_inputs: X})
		if self.pca is not None:
			z_mu = self.pca.transform(z_mu)
		return z_mu

	def generate(self, sess, z_mu=None):
		if z_mu is None:
			z_mu = np.random.normal(size=[100, self.nlatent])
		if self.pca is not None:
			z_mu = self.pca.inverse_transform(z_mu)
		return sess.run(self.vae_outputs, feed_dict={self.z: z_mu})

	def get_focus(self, sess, X):
		regen = self.generate(sess, self.transform(sess, X))
		return obj_atten(regen)

A, B = 6.5, 15.0
def focus_actfn(x):
	return 1.0/(1 + B*np.exp(-A * x))

def obj_atten(imgs):
	atten_imgs = np.zeros(imgs.shape)
	for k, img in enumerate(imgs):
		mx, my = img.shape[0]/2.0, img.shape[1]/2.0
		lkhood = 0.0;
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				d = np.sqrt((mx-i)**2 + (my-j)**2)
				lkhood = img[i][j] * d - (1.0 - img[i][j]) * d
		# atten_imgs.append(1.0 / (1 + np.exp(3*(0.5-img if lkhood <= 0.0 else
  #       img-0.5))))
  		print k, "-> likelihood:", lkhood
		atten_imgs[k, ...] = focus_actfn(1.0-img if lkhood >= 0.0 else img)
		# atten_imgs[k, ...] = focus_actfn(img)
	return atten_imgs

def np_rgb2grey(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])[..., np.newaxis]

FOCUS_BSIZE = 1000
def rgbfocus(vae, sess, imgs):
    nimg = imgs.shape[0]
    fimgs = np.zeros((FOCUS_BSIZE, imgs.shape[1], imgs.shape[2], 3))
    focus_out = np.zeros((FOCUS_BSIZE, imgs.shape[1], imgs.shape[2], 1))
    for i in range(0, nimg, FOCUS_BSIZE):
    	print i
    	gimgs = np_rgb2grey(imgs[i:min(nimg, i+FOCUS_BSIZE)])
        focus = vae.get_focus(sess, gimgs)
        fimgs = np.zeros((FOCUS_BSIZE, imgs.shape[1], imgs.shape[2], 3))
        print focus.shape, fimgs.shape, imgs.shape
        for k in range(3):
        	fimgs[i:min(nimg, i+FOCUS_BSIZE), :, :, k] = \
        		imgs[i:min(nimg, i+FOCUS_BSIZE), :, :, k] * focus[:, :, :, 0]
        focus_out[i:min(nimg, i+FOCUS_BSIZE), ...] = focus

    print np.max(imgs), np.max(fimgs)
    return fimgs, focus_out

def test_focus(sess, mdb, vae, batch_rsq=5):
	### reconstruction by exact mean ###
	batch_img, batch_label = mdb.get_batch(batch_rsq**2, mode='test', size=vae.in_size, cmap='rgb')
	focus_img, focus_f = rgbfocus(vae, sess, batch_img)

	### plot results ###
	plt.figure()
	plt.suptitle('Dataset')
	for i in range(batch_img.shape[0]):
		plt.subplot(batch_rsq, batch_rsq, i+1)
		plt.imshow(batch_img[i, ...])
		plt.axis('off')
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()

	plt.figure()
	plt.suptitle('Focus Filter')
	for i in range(batch_img.shape[0]):
		plt.subplot(batch_rsq, batch_rsq, i+1)
		plt.imshow(focus_f[i, ..., 0], cmap='gray')
		plt.axis('off')
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()

	plt.figure()
	plt.suptitle('Focused Dataset')
	for i in range(batch_img.shape[0]):
		plt.subplot(batch_rsq, batch_rsq, i+1)
		plt.imshow(focus_img[i, ...])
		plt.axis('off')
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()

def test_reconstruct(sess, mdb, vae, batch_rsq=5):
	### reconstruction by exact mean ###
	batch_img, batch_label = mdb.get_batch(batch_rsq**2, mode='test', size=vae.in_size, cmap='grey')
	mean_img = vae.transform(sess, batch_img)
	recon_img = vae.generate(sess, mean_img)
	# recon_img = vae.generate(sess, None)

	### plot results ###
	plt.figure()
	plt.suptitle('Dataset')
	for i in range(batch_img.shape[0]):
		plt.subplot(batch_rsq, batch_rsq, i+1)
		plt.imshow(batch_img[i, ..., 0])
		plt.axis('off')
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()
	plt.figure()
	plt.suptitle('Reconstructed Dataset')
	for i in range(batch_img.shape[0]):
		plt.subplot(batch_rsq, batch_rsq, i+1)
		plt.imshow(recon_img[i, ..., 0])
		plt.axis('off')
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()

if __name__ == '__main__':
	# dataset = mnist_db()
	# dataset = image_db('wrenches')
	# dataset = image_db('springs/all')
	labels = cifar10_read_label("../trainLabels.csv")
	dataset = image_db('../train', train_portion=0.8, seed=42)
	dataset.transform_label(lambda x: labels.i2n(x))

	vae = Vae(name="2d_vae_0", in_size=[32, 32], \
		cnvlf=[32, 32, 16], kernel_size=[[5,5], [5,5], [3, 3]], strides=[2, 2,
        2], \
		d_cnvlf=[32, 32, 1], d_kernel_size=[[5,5], [5,5], [3, 3]],
        d_strides=[2, 2, 2], \
		nlatent=16, lr=0.001)

	sess = tf.InteractiveSession()
	# with tf.Session() as sess:
	sess.run(vae.init)

	### check if trained model exists ###
	vae.restore(sess)

	### training ###
	if not vae.is_trained:
		vae.train(sess, dataset, batch_size=200, epoch_num=300)
	# vae.train(sess, dataset, batch_size=200, epoch_num=300)

	### swrite summary ###
	vae.write_summary(sess, filename="viz/vae/"+vae.name_scope)

	### testing ###
	# test_reconstruct(sess, dataset, vae, batch_rsq=10)
	test_focus(sess, dataset, vae, batch_rsq=3)
	plt.show()
	plt.close()
	
	# batch_img, batch_label = vae.dataset.get_batch(1, mode='test', size=[32, 32])
	# latents = vae.vae_latents.eval(feed_dict={vae.vae_inputs:batch_img})
	# print latents
