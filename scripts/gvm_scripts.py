import pandas as pd
import numpy as np
import os
import csv
import pickle
import h5sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, vstack, hstack
from itertools import compress

def write_gvm(gvm, output_fname, fmt='h5'):
	if fmt == 'pd':
		if type(gvm) != pd.DataFrame:
			temp = pd.DataFrame(gvm['gvm'].todense())
			if ('idx' not in gvm) or np.all(np.array(gvm['idx'] == None)):
				gvm['idx'] = np.array(range(gvm['gvm'].shape[0]))
			if ('col' not in gvm) or np.all(np.array(gvm['col'] == None)):
				gvm['col'] = np.array(range(gvm['gvm'].shape[1]))
			temp.index = gvm['idx']
			temp.columns = gvm['col']
			gvm = temp

		gvm = gvm.replace(to_replace=False, value='')
		gvm.to_csv(output_fname, sep='\t')

	elif fmt == 'h5':
		if type(gvm) == pd.DataFrame:
			gvm = {'gvm': csc_matrix(gvm.values),
				   'idx': gvm.index.values,
				   'col': gvm.columns.values}

		if ('idx' not in gvm) or np.all(np.array(gvm['idx'] == None)):
			gvm['idx'] = np.array(range(gvm['gvm'].shape[0]))
		if ('col' not in gvm) or np.all(np.array(gvm['col'] == None)):
			gvm['col'] = np.array(range(gvm['gvm'].shape[1]))

		if not ( np.all([isinstance(i, int) for i in gvm['idx']]) or
				 np.all([isinstance(i, float) for i in gvm['idx']]) ):
			try: gvm['idx'] = np.array([i.encode('utf-8','ignore') for i in gvm['idx']], dtype=np.string_)
			except AttributeError: gvm['idx'] = np.array(gvm['idx'], dtype=np.string_)
		if not ( np.all([isinstance(i, int) for i in gvm['col']]) or
				 np.all([isinstance(i, float) for i in gvm['col']]) ):
			try: gvm['col'] = np.array([i.encode('utf-8','ignore') for i in gvm['col']], dtype=np.string_)
			except AttributeError: gvm['idx'] = np.array(gvm['idx'], dtype=np.string_)

		with h5sparse.File(output_fname, 'w') as h:
			h.create_dataset('gvm', data=gvm['gvm'])
			h.create_dataset('idx', data=np.array(gvm['idx']))
			h.create_dataset('col', data=np.array(gvm['col']))

	else: raise ValueError('unrecognized format %s'%fmt)

def open_gvm(fname):
	'''
	Returns a gvm obtained from the input file.
	fname : str
		The name of the file to be obtained as a gvm.
	'''

	with h5sparse.File(fname, 'r') as h:
		data = h['gvm'][()]
		if 'idx' in h:
			idx = h['idx'][()]
			try: idx = [i.decode('utf-8','ignore') for i in idx]
			except: pass
		else: idx = None

		if 'col' in h:
			col = h['col'][()]
			try: col = [i.decode('utf-8','ignore') for i in col]
			except: pass
		else: col = None
	return {'gvm': data, 'idx': np.array(idx), 'col': np.array(col)}

def get_gvm_size(fname):
	with h5sparse.File(fname, 'r') as h:
		return h['gvm'].shape

def file_exists(fname, warning=True):
	'''
	Checks if a file exists or not. Returns True or False. Prints a statement if False.
	fname : str
		The name of the file to check for existence.
	'''
	if type(fname) not in [str, bytes, os.PathLike]: return False
	if os.path.isfile(fname):
		if warning: print('\t', fname, 'has already been created.')
		return True
	else: return False

def format_gene_names(names, pct_commas_threshold=.1, verbose=True):

	# Uppercase and remove whitespace.
	names = np.char.strip(np.char.upper(np.array(names, dtype='str')))

	# Remove suffixes, if > 10% of gene names have commas.
	n_commas = sum([',' in i for i in names])
	if n_commas > len(names) * pct_commas_threshold:
		if verbose: print('\t %d of %d gene names have commas. Removing the suffixes.'%(n_commas, len(names)))
		names = np.array([i.partition(',')[0] for i in names])
	names = np.where(names == '', 'EMPTY GENE NAME', names)
	return names

def transpose_gvm(gvm):		
	gvm['gvm'] = gvm['gvm'].transpose()
	gvm['idx'], gvm['col'] = (gvm['col'], gvm['idx'])

	return gvm

def merge_dup_gene_names(gvm, verbose=False):

	# TODO: edit code so transpose is not necessary.
	gvm = transpose_gvm(gvm)

	idx = gvm['idx']
	data = gvm['gvm']

	unique_genes, counts = np.unique(idx, return_counts=True)
	dups = unique_genes[counts > 1]
	if len(dups) == 0: return transpose_gvm(gvm)
	if verbose: print('%d genes have duplicates.'%len(dups))

	gene_id = {g:i for i,g in enumerate(unique_genes)}
	# Add duplicate gene-rows together.
	Mat_data = np.ones(shape=(len(idx),))
	Mat_xy = ([gene_id[g] for g in idx], range(len(idx)))
	Mat = coo_matrix((Mat_data, Mat_xy), shape=(len(gene_id), len(idx)))
	data = Mat * data
	# Replace all nonzero entries with one.
	# Overall, this is equivalent to taking the union.
	data.data = np.ones(shape=(len(data.data),), dtype=int)

	gvm['idx'] = unique_genes
	gvm['gvm']	 = csc_matrix(data, dtype=bool)

	gvm = transpose_gvm(gvm)

	return gvm

def get_genesetlist(item, item_type):
	'''
	Returns a "geneset list" representation of the input item.
	This is a pandas.Series where the values are the annotations' geneset lists
	and the index is the annotations, e.g.
		(Index)			(Value)
		CHEMBL406270	[MSRA, TIP1, ...]
		CHEMBL279107	[ENP1, ...]
		CHEMBL47181		[TIP1, ...]
		...				...
	The input item can be a gmt file name, gvm, gvm file name, or interaction list.
	item : str (gmt file name or gvm file name), pd.DataFrame (gvm), or pd.Series (interaction list)
		The item to obtain as a geneset list.
	item_type : str
		Indicates the type of `item`. One of: "gmt_fname", "gvm", "gvm_fname", or "interactionlist"/"ilist".
	'''
	if item_type == 'gmt_fname':
		gmt_fname = item
		with open(gmt_fname, 'r') as f:
			reader = csv.reader(f, delimiter = '\t')
			# This will remove everything after the first comma!
			d = {row[0]:sorted(set([str(g).split(',')[0] for g in row[2:] if g != ''])) for row in reader}
			return pd.Series(d).sort_index()

	elif item_type in ['interactionlist', 'ilist']:
		return item.groupby('annotation')['gene'].apply(set).apply(sorted).sort_index()

	else: raise ValueError('unknown item type ' + item_type)

def convert_genesetlist(gslist, to, output_fname = None, verbose = False):
	'''
	Converts an input geneset list into another representation: gmt or gvm. Returns it.
	If `to == gmt` and an output file name is given, it will save the results to `output_fname`.
	If `output_fname` already exists, then the results saved to that file will be used.
	gslist : pandas.Series
		The geneset list to be converted.
	to : str
		Either 'gmt' or 'gvm'
	output_fname : str
		The name of the file to save the results to.
	verbose : bool
		Control the frequency of print statements used when converting to gvm.

	'''
	if verbose: print('obtaining ' + output_fname)
	if to == 'gmt':
		#Create the gmt.
		gmt = [[annot] + [''] + genes for (annot,genes) in zip(gslist.index, gslist.values)]
		#Save it to the file if it does not exist yet.
		if output_fname is not None:
			if not file_exists(output_fname):
				with open(output_fname, 'w', newline='') as f:
					writer = csv.writer(f, delimiter='\t')
					for geneset in gmt: writer.writerow(geneset)
		return gmt

	elif to == 'gvm_h5':
		#If the gvm file already exists, load it and return it.
		if file_exists(output_fname):
			return open_gvm(output_fname)
		elif file_exists(output_fname.replace('.gvm','gvm.pkl')):
			return open_gvm(output_fname.replace('.gvm','gvm.pkl'))

		#Otherwise, create it.
		all_genes_set = {item for sublist in gslist for item in sublist}
		all_genes = pd.Series(sorted(all_genes_set))
		gslist = gslist.apply(set)
		gvm = [csr_matrix(all_genes.isin(gs), dtype=bool) for gs in gslist]
		gvm = vstack(gvm)

		gvm = {'gvm':gvm, 'idx':gslist.index, 'col':all_genes}
		if output_fname is not None: write_gvm(gvm, output_fname, fmt='h5')
		return gvm

	else: raise ValueError('The desired representation (`to`) is unsupported: ' + to)

def format_gvm_h5(gvm_fname, all_genes, output_fname, min_gs_size = 3, max_gs_loss=.5,
				 verbose=True, overwrite=True, return_value='summary'):

	if not overwrite:
		if os.path.isfile(output_fname):
			print('%s already created.' %output_fname)
			with h5sparse.File(output_fname, 'r') as h:
				if return_value == 'gvm': return h['gvm'][()]
				elif return_value == 'shape': return h['gvm'].shape

	if verbose: print(gvm_fname + ':')
	gvm = open_gvm(gvm_fname)

	n_labels, n_genes = gvm['gvm'].shape
	genes_per_label = np.squeeze(np.asarray(gvm['gvm'].sum(axis=1)))

	# Capitalize. Remove suffixes, if > 10% of gene names have commas.
	gvm['col'] = np.array(format_gene_names(gvm['col']))

	# Replace gene index with gene IDs.
	# Mark genes not in `all_genes` for deletion.
	gene_dict = dict(zip(all_genes.index, all_genes['gene id']))
	gvm['col'] = np.array([gene_dict.get(i, -1) for i in gvm['col']])
	col_keep = np.nonzero(gvm['col'] != -1)[0]
            
	# Drop rare and non-identifiable genes.
	gvm['gvm'] = gvm['gvm'][:,col_keep]
	gvm['col'] = gvm['col'][col_keep]
	n_genes_lost = n_genes - gvm['gvm'].shape[1]
	if verbose: print('\t%d out of %d genes were removed.' % (n_genes_lost, n_genes))

	#Take union for genes mapped onto one another.
	genes_mapped_same_symbol = len(gvm['col']) - len(set(gvm['col']))
	if verbose: print('\t%d genes were mapped onto a pre-existing gene.' % (genes_mapped_same_symbol))
	gvm = merge_dup_gene_names(gvm)

	# Drop labels which have less than min_gs_size genes in them.
	# Also, drop labels which lost a proportion > max_gs_loss of their genes.
	genes_per_label_change_prop = np.where(genes_per_label == 0, 0,
		( genes_per_label - np.squeeze(np.asarray(gvm['gvm'].sum(axis=1))) ) / genes_per_label)
	row_keep = np.nonzero( (
		np.squeeze(np.asarray((gvm['gvm'].sum(axis=1)))) >= min_gs_size) & (
		genes_per_label_change_prop <= max_gs_loss) )[0]
	gvm['gvm'] = gvm['gvm'][row_keep,:]
	gvm['idx'] = gvm['idx'][row_keep]
	n_labels_lost = n_labels - gvm['gvm'].shape[0]
	if verbose: print('\t%d out of %d labels were removed.' % (n_labels_lost, n_labels))

	#Re-format gene vectors to standard size and order.
	gvm['col'] = gvm['col'].astype('int64')
	all_valid_genes = all_genes['gene id'][all_genes['gene id'] != -1]
	missing_genes = list(set(all_valid_genes) - set(gvm['col']))
	missing_cols = csr_matrix( np.full(fill_value=False, shape=(gvm['gvm'].shape[0], len(missing_genes))) )
	gvm['gvm'] = csr_matrix(hstack([gvm['gvm'], missing_cols]))
	gvm['col'] = np.append(gvm['col'], np.array(missing_genes))
	col_sorting = np.argsort(gvm['col'])
	gvm['gvm'] = gvm['gvm'][:,col_sorting]
	gvm['col'] = gvm['col'][col_sorting]
	
	write_gvm(gvm, output_fname, fmt='h5')

	if return_value == 'gvm':
		return open_gvm(output_fname)
	elif return_value == 'summary':
		out = {'formatted gvm shape': get_gvm_size(output_fname), 
			   'genes removed': n_genes_lost,
			   'genes mapped to same symbol': genes_mapped_same_symbol,
			   'labels removed': n_labels_lost}
		return out