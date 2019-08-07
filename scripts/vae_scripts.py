import numpy as np
import h5py
import h5sparse

import scipy.sparse as sparse
from scipy.stats import norm

from sklearn.preprocessing import normalize

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Layer, Lambda, Multiply, Add, Dropout
from keras.utils import Sequence
from keras.optimizers import Adam, Adamax
from keras.utils.vis_utils import model_to_dot, plot_model
from keras import backend as K
from keras.regularizers import l1
from keras.callbacks import Callback
from time import sleep

def partition(all_labs, method='lib holdout', holdout=None, pct=.05, seed=None):
	libs = list(set(all_labs['lib name']))

	if method=='lib holdout':
		if holdout is None:
			np.random.seed(seed)
			holdout = np.random.choice(libs)
			print('Holdout library: %s' %holdout)
		if type(holdout) == str: holdout = [holdout]
		test = np.array(all_labs['lib name']) == holdout

	elif method=='balanced pct':
		test = np.full((all_labs.shape[0],), False)
		for l in libs:
			labs = all_labs['lib name'] == l
			n_true = int(np.floor(pct*sum(labs)))
			test_choices = [True]*n_true + [False]*int(sum(labs) - n_true)
			np.random.seed(seed)
			np.random.shuffle(test_choices)
			test[labs] = test_choices

	elif method=='pct':
		np.random.seed(seed)
		test = p.random.choice(a=(False, True), size=(all_labs.shape[0],), p=(pct, 1-pct))

	part = {'train': all_labs.index.values[~test], 'test': all_labs.index.values[test]}
	return part

class GeneVec_Generator(Sequence):
	'Generates batches of data for Keras'
	#Assumes individual gvms fit in memory.
	# Adaptation of:
	# ttps://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
	# Afshine Amidi and Shervine Amidi

	def __init__(self, gvm_part_fname, gvm_path, indices=None,
			batch_size=128, shuffle = True, verbose = True, norm=False):

		'Initialization'
		if gvm_path is None:
			with h5sparse.File(gvm_part_fname, 'r') as h:
					self.GVD = h[()]
		else:
			with h5sparse.File(gvm_part_fname, 'r') as h:
				self.GVD = h[gvm_path][()]
		# If no indices are specified, use all of them.
		if indices is None:
			with h5sparse.File(gvm_part_fname, 'r') as h:
				indices = list(range(self.GVD.shape[0]))
		self.indices = list(map(int, indices))
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.verbose = verbose
		self.normalize = norm

		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.ceil(len(self.indices) / self.batch_size))

	def __getitem__(self, batch_index):
		'Generate one batch of data'
		label_indices = self.indices[
			(batch_index*self.batch_size):(
			(batch_index+1)*self.batch_size)]
		X = self.GVD[label_indices,:].todense() # no todense required.
		if self.normalize: X = normalize(X)
		return X, X

	def on_epoch_end(self):
		'Updates indices after each epoch'
		if self.shuffle: np.random.shuffle(self.indices)

def compute_kernel(x, y, sigma_sqr):
	x_size = K.shape(x)[0]
	y_size = K.shape(y)[0]
	dim = K.shape(x)[1]
	tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
	tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
	#return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim * sigma_sqr, 'float64'))
	return K.exp(-sigma_sqr * K.mean(K.square(tiled_x - tiled_y), axis=2))
	# n samples

def compute_mmd(x, y, sigma_sqr=1.):
	x_kernel = compute_kernel(x, x, sigma_sqr)
	y_kernel = compute_kernel(y, y, sigma_sqr)
	xy_kernel = compute_kernel(x, y, sigma_sqr)
	return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
	# single value

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mu + sqrt(var)*eps
def sampling(args):
	"""Reparameterization trick by sampling from an isotropic unit Gaussian.
	# Arguments
		args (tensor): mean and log of variance of Q(z|X)

	# Returns
		z (tensor): sampled latent vector
	"""

	z_mu, z_log_var = args
	batch = K.shape(z_mu)[0]
	dim = K.int_shape(z_mu)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mu + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, middle_dim=1000, latent_dim=100, regularizer='KL',
	epsilon_std=1.0, h_act = 'relu', variational=False, DROPOUT_RATE=0,
	epochs=30, BETA_START=0, optimizer='Adamax', LEARNING_RATE=.001, sparse=None,
	POSCLASS_ETA=None, MMD_SIGMA_SQR=None, MMD_SCALE=None):
	# http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

	opt_dict = {'Adam': Adam, 'Adamax': Adamax}
	optimizer = opt_dict[optimizer]

	if variational:
		if sparse is not None:
			print('Warning: sparsity constraint not implemented for VAE. Ignoring sparsity constraint.')
		act_reg = None
	else:
		if sparse is not None:
			act_reg = l1(sparse)
		else: 
			act_reg = None

	if regularizer == 'MMD':
		if MMD_SIGMA_SQR is None: MMD_SIGMA_SQR = 2 / latent_dim
		if MMD_SCALE is None: MMD_SCALE = input_dim / latent_dim
			
	# x
	x = Input(shape=(input_dim,), name='x')

	# encoder hidden layer
	if DROPOUT_RATE != 0:
		xd = Dropout(rate=DROPOUT_RATE, name='drop_input')(x)
		h_enc = Dense(middle_dim, activation=h_act, name='hidden_enc')(xd)
	else:
		h_enc = Dense(middle_dim, activation=h_act, name='hidden_enc')(x)

	# latent space
	if variational:
		z_mu = Dense(latent_dim, name='mu')(h_enc)
		z_log_var = Dense(latent_dim, name='log_var')(h_enc)
		z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mu, z_log_var])
	else:
		z = Dense(latent_dim, activity_regularizer=act_reg, name='z')(h_enc)

	# decoder hidden layer and xhat
	dec = Sequential([
		Dense(middle_dim, input_dim=latent_dim, activation=h_act),
		Dense(input_dim, activation='sigmoid')
	], name='dec')
	xhat = dec(z)

	vae = Model(inputs=x, outputs=xhat, name='vae')

	def bce_loss(x, xhat):
		return K.mean(K.binary_crossentropy(x, xhat), axis=-1) * input_dim
		# K.mean(K.binary_crossentropy(x, xhat), axis=-1) === keras.losses.binary_crosentropy(x, xhat)
	# n_samples

	def bce_eta_loss(x, xhat):
		return K.mean(K.binary_crossentropy(x, xhat) * (1 + POSCLASS_ETA * x), axis=-1) * input_dim

	def kl_loss(x, xhat):
		return -.5 * K.sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=1)
	# n_genes size
	def mmd_loss(x, xhat):
		normal_sample = K.random_normal(shape=K.shape(z), dtype='float32')
		mmd = compute_mmd(x=normal_sample, y=z, sigma_sqr=MMD_SIGMA_SQR)
		return(mmd * MMD_SCALE)

	beta = K.variable(value=BETA_START)
	def m_beta(x, xhat):
		return beta

	if POSCLASS_ETA is None: reconst_loss = bce_loss
	else: reconst_loss = bce_eta_loss

	if variational:
		if regularizer == 'KL': reg_loss = kl_loss
		elif regularizer == 'MMD': reg_loss = mmd_loss

	def joint_loss(x, xhat):
		return K.mean(reconst_loss(x, xhat) + beta*reg_loss(x, xhat))

	if variational:
		loss = joint_loss
		metrics = [reconst_loss, reg_loss, m_beta]
	else:
		loss = reconst_loss
		metrics = [reconst_loss]
	vae.compile(optimizer=optimizer(lr=LEARNING_RATE), loss=loss, metrics=metrics)

	if variational:
		enc = Model(x, z_mu)
	else:
		enc = Model(x, z)

	out = {'vae': vae, 'enc': enc, 'dec': dec, 'beta': beta}
	return out