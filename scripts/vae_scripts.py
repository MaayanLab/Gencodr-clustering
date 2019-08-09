import numpy as np
import h5py
import h5sparse

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
	'''
	partitions all the samples into a test and train set. 

	all_labs (pd.DataFrame): table of sample labels (i.e. annotations) made by `scripts/0_merge_libraries.ipynb`
	method (str): one of
		'lib holdout': uses one library for the test set, and the rest for the train set
		'balanced pct': uses `pct` of each library for the test set, and the rest for the train set
		'pct'
	holdout (str): Optional. The library to holdout if method=='lib holdout'. If None, chosen randomly.
	pct (float): 0 to 1. Proportion to reserve for the test set if method=='balanced pct'
	seed (int): seed for RNG
	'''
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

	else: raise NotImplementedError('method should be "lib holdout" or "balanced pct"')

	part = {'train': all_labs.index.values[~test], 'test': all_labs.index.values[test]}
	return part

class GeneVec_Generator(Sequence):
	'''
	Generates batches of data for Keras
	Assumes individual gvms fit in memory.
	Source: ttps://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
	Afshine Amidi and Shervine Amidi
	'''

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
		X = self.GVD[label_indices,:].todense() # no todense required?
		if self.normalize: X = normalize(X)
		return X, X

	def on_epoch_end(self):
		'Updates indices after each epoch'
		if self.shuffle: np.random.shuffle(self.indices)

def compute_kernel(x, y, sigma_sqr):
	'''Helper function for `compute_mmd`'''
	x_size = K.shape(x)[0]
	y_size = K.shape(y)[0]
	dim = K.shape(x)[1]
	tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
	tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
	#return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim * sigma_sqr, 'float64'))
	return K.exp(-sigma_sqr * K.mean(K.square(tiled_x - tiled_y), axis=2))
	# n samples

def compute_mmd(x, y, sigma_sqr=1.):
	'''
	Computes the Maximum Mean-Discrepancy.
	(It runs, but I haven't confirmed this code is correct.)
	Source: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
	Shengjia Zhao

	Parameters:
		x (tensor)
		y (tensor)
	'''
	x_kernel = compute_kernel(x, x, sigma_sqr)
	y_kernel = compute_kernel(y, y, sigma_sqr)
	xy_kernel = compute_kernel(x, y, sigma_sqr)
	return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
def sampling(args):
	'''
	Reparameterization trick by sampling from eps = N(0, I).
	z = z_mu + sqrt(var)*eps

	Parameters:
		args (tensor): mean and log of variance of Q(z|X)

	Returns:
		z (tensor): sampled latent vector
	'''

	z_mu, z_log_var = args
	batch = K.shape(z_mu)[0]
	dim = K.int_shape(z_mu)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mu + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, middle_dim=1000, latent_dim=100, regularizer='KL',
	h_act = 'relu', variational=False, DROPOUT_RATE=0,
	epochs=30, BETA_START=0, optimizer='Adamax', LEARNING_RATE=.001, sparse=None,
	POSCLASS_ETA=None, MMD_SIGMA_SQR=None, MMD_SCALE=None):
	'''
	Wrapper for building the Keras (V)AE model.
	# http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
	
	Parameters:
		input_dim (int): The dimensionality of the input
		middle_dim (int): The dim. of the intermediate dimensions of the encoder and decoder
		latent_dim (int): The dim. of the latent space
		regularizer (str): 'MMD' or 'KL'. (for the AE, is computed but not added to loss function.)
		h_act (str): Keras activation. https://keras.io/activations/
		variational (bool): VAE?
		DROPOUT_RATE (float): 0 to 1. If > 0, adds a dropout layer with this rate to the input layer.
		epochs (int): # epochs to run for (unless EarlyStopping occurs)
		BETA_START (int): The initial multiplier for the regularizer loss (if VAE).
		optimizer (str): Must be 'Adamax' or 'Adam': the optimizer used to train the autoencoder.
		LEARNING_RATE (float): the learning rate
		sparse (bool): sparse? if so, adds a l1(sparse) activity regularizer to the latent layer.
		POSCLASS_ETA (float): the loss function for indices == 0 is multiplied by (1 + POSCLASS_ETA).
		MMD_SIGMA_SQR: the sigma squared value for the MMD regularizer.
		MMD_SCALE: the scale value for the MMD regularizer.
	'''
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

class PatchedModelCheckpoint(Callback):
	"""
	https://github.com/keras-team/keras/issues/11101#issuecomment-502023233
	Save the model after every epoch.
	`filepath` can contain named formatting options,
	which will be filled with the values of `epoch` and
	keys in `logs` (passed in `on_epoch_end`).
	For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
	then the model checkpoints will be saved with the epoch number and
	the validation loss in the filename.
	# Arguments
		filepath: string, path to save the model file.
		monitor: quantity to monitor.
		verbose: verbosity mode, 0 or 1.
		save_best_only: if `save_best_only=True`,
			the latest best model according to
			the quantity monitored will not be overwritten.
		save_weights_only: if True, then only the model's weights will be
			saved (`model.save_weights(filepath)`), else the full model
			is saved (`model.save(filepath)`).
		mode: one of {auto, min, max}.
			If `save_best_only=True`, the decision
			to overwrite the current save file is made
			based on either the maximization or the
			minimization of the monitored quantity. For `val_acc`,
			this should be `max`, for `val_loss` this should
			be `min`, etc. In `auto` mode, the direction is
			automatically inferred from the name of the monitored quantity.
		period: Interval (number of epochs) between checkpoints.
	"""

	def __init__(self, filepath, monitor='val_loss', verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1):
		super(PatchedModelCheckpoint, self).__init__()
		self.monitor = monitor
		self.verbose = verbose
		self.filepath = filepath
		self.save_best_only = save_best_only
		self.save_weights_only = save_weights_only
		self.period = period
		self.epochs_since_last_save = 0

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpoint mode %s is unknown, '
						  'fallback to auto mode.' % (mode),
						  RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
				self.monitor_op = np.greater
				self.best = -np.Inf
			else:
				self.monitor_op = np.less
				self.best = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch=epoch + 1, **logs)
			if self.save_best_only:
				current = logs.get(self.monitor)
				if np.isnan(current): raise ValueError('nan loss.')
				if current is None:
					warnings.warn('Can save best model only with %s available, '
								  'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
						if self.verbose > 0:
							print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
								  ' saving model to %s'
								  % (epoch + 1, self.monitor, self.best,
									 current, filepath))
						self.best = current

						saved_correctly = False
						while not saved_correctly:
							try:
								if self.save_weights_only:
									self.model.save_weights(filepath, overwrite=True)
								else:
									self.model.save(filepath, overwrite=True)
								saved_correctly = True
							except Exception as error:
								print('Error while trying to save the model: {}.\nTrying again...'.format(error))
								sleep(3)
					else:
						if self.verbose > 0:
							print('\nEpoch %05d: %s did not improve from %0.5f' %
								  (epoch + 1, self.monitor, self.best))
			else:
				if self.verbose > 0:
					print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
				saved_correctly = False
				while not saved_correctly:
					try:
						if self.save_weights_only:
							self.model.save_weights(filepath, overwrite=True)
						else:
							self.model.save(filepath, overwrite=True)
						saved_correctly = True
					except Exception as error:
						print('Error while trying to save the model: {}.\nTrying again...'.format(error))
						sleep(5)