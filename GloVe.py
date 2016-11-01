#### Libraries
# Standard Library
import collections
from collections import Counter
import itertools 
import re

# Third Party Library
import numpy as np
from scipy import sparse
import theano
import theano.tensor as T

def cooccur_matrix(corpus, window_size=7, min_count=None):
	"""Takes the corpus and creates a list where each entry is the word
	appearing in the corpus. The words are then transformed to lower 
	case where a list is created containing the unique words in the 
	corpus. Each word is then indexed and the lowered case corpus is 
	transformed using the word index. This is followed by creating a 
	context window around each word which is used to generate a 
	cooccurance matrix.
	"""
	
	vocab = corpus.split() 
	vocab = [element.lower() for element in vocab]
	tokens = list(set(vocab)) # Gets the unique words in the corpus
	token_size = len(tokens)

	cooccurrences = sparse.coo_matrix((len(tokens)+1, len(tokens)+1), 
					   dtype=theano.config.floatX)

	word_to_index = {words: i for i, words in enumerate(tokens)}
	# index_to_word = {index: word for word, index in word_to_index.items()}

	# Creates a window centered around each word
	win = 2 * window_size + 1
	l = list(map(lambda x: word_to_index[x], vocab))
	lpadded = win // 2 * [token_size] + l + win // 2 * [token_size] 

	out = [lpadded[i:(i + win)] for i in range(len(l))]
	assert len(out) == len(l)

	# Creates the cooccurrence matrix
	for windows in out:
		r = [windows[window_size] for i in range(win-1)]
		c = windows[:window_size] + windows[window_size+1:]
		d = np.ones((win-1,)) 

		sparse_temp = sparse.coo_matrix((d, (r, c)), 
						shape=(len(tokens)+1, len(tokens)+1),
					        dtype=theano.config.floatX)

		cooccurrences += sparse_temp

	cooccurrences = cooccurrences[:-1, :-1] # Removes the padding row and column


	# TODO remove instance of that word from dictionary when the row and
	#	   column are removed. 

	# TODO Find a way of deleting column and rows that fail to meet min
	#	   threshold criteria without using cooccurrences.toarray().
	# 	   This would take advantage of the sparcity structure of 
	# 	   cooccurrences

	# if min_count is not None and min_count > 0:
	# 	cooccurrences = cooccurrences.toarray()
	# 	mask = cooccurrences >= min_count
	# 	cooccurrences = cooccurrences[mask.any(1)][:,mask.any(0)]

	return cooccurrences

def train(cooc_matrix, x_max=3, alpha=0.75, learning_rate=0.05):
	values = list(cooc_matrix.data)

	cooc_matrix = np.transpose(np.nonzero(cooc_matrix))
	main_row = cooc_matrix[:, 0]
	context_row = cooc_matrix[:, 1]

	main_matrix = np.zeros((cooc_matrix.shape[0], cooc_matrix.shape[1]+1))
	main_matrix[:, 0] = main_row
	main_matrix[:, 1] = context_row
	main_matrix[:, 2] = values

	del(cooc_matrix)

	word_count = len(word_to_vec)
	vector_sums = np.zeros((2*word_count, word_to_vec[0][0].shape[0]+1)) # upper-half of matrix is main, lower_half is context
	total_cost = 0

	for tuples in main_matrix:
		weight = (tuples[2]/x_max)**alpha if tuples[2] < x_max else 1
		main = int(tuples[0])
		context = int(tuples[1])

		inner_cost = np.dot(word_to_vec[main][0], word_to_vec[context][1])\
				   + word_to_vec[main][2] + word_to_vec[context][3]\
				   - np.log(tuples[2])
		total_cost += 0.5 * weight * inner_cost**2

		# Fills the vector_sums matrix with the sums in relation to the
		# main and context word. `vector_sums` will then be used in the 
		# gradient to compute SGD. 
		vector_sums[main, :-1] += weight * inner_cost * word_to_vec[context][1]
		vector_sums[context+word_count, :-1] += weight * inner_cost *\
										   		word_to_vec[main][0]
		vector_sums[main, -1] += weight * inner_cost * word_to_vec[context][3]				   		
		vector_sums[context+word_count, -1] += weight * inner_cost *\
											   word_to_vec[main][2]


	# TODO Change SGD to use RMSPROP
	# Vanilla Gradient Descent
	for i in range(word_count):
		word_to_vec[i][0] =- learning_rate * vector_sums[i, 0:3]
		word_to_vec[i][1] =- learning_rate * vector_sums[i+word_count, 0:3]
		word_to_vec[i][2] =- learning_rate * vector_sums[i, 3]
		word_to_vec[i][3] =- learning_rate * vector_sums[i+word_count, 3]


if __name__ == '__main__':
	file = '...\\test.txt'
	text = open(file, 'r').read()

	temp = cooccur_matrix(text)
	train(temp)

