#### Libraries
# Standard Library
import collections
from collections import Counter
import itertools 
import re

# Third Party Library
import matplotlib.pyplot as plt
import nltk
import numpy as np
from scipy import sparse

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
									  dtype=np.float32)

	word_to_index = {words: i for i, words in enumerate(tokens)}
	# index_to_word = {index: word for word, index in word_to_index.items()}

	# Creates a window around each word
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
										shape=(len(tokens)+1, len(tokens)+1))

		cooccurrences += sparse_temp

	cooccurrences = cooccurrences[:-1, :-1] # Removes the padding row and column

	# If `min_count` is not `None` and `min_count` > 0 then we will 
	# remove the instance of that word from the `cooccurrences` matrix.
	# Since the cooccurrence matrix is symmetric then the row and column
	# contaning that word will be removed. 

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

if __name__ == '__main__':
	file = 'C:\\Users\\lukas\\Documents\\NLP\\Word_Embedding\\Basketball_test.txt'
	text = open(file, 'r').read()