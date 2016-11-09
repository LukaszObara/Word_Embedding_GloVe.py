#### Libraries
# Standard Library
import pickle

# Third Party Libraries
import numpy as np
from scipy import sparse

class GloVe(object):
	def __init__(self, corpus, window_size=3, vector_length=2):
		self.corpus = corpus
		self.window_size = window_size
		self.vec_length = vector_length
		# The bottom values are initialized after a corpus has been fed 
		self.token_size = 0
		self.word_to_index = None
		self.word_to_vec = None
		self.cache_matrix = None 
		# Due to the variable nature of text documents in relation to 
		# the number of words The value of cache matrix` will change 
		# after `cooccur_matrix()` is called. The result will be a 
		# matrix of size:
		#
		#          2*unique_word_count x (vector_length + 1)
		#
		# where the +1 is to account for the bias. The upper half of the
		# matrix will be assigned to the main word vectors, whereas the 
		# bottom half will be assigned to the context word vectors. 

	def vocabulary(self):
		vocab = self.corpus.split() 
		vocab = [element.lower() for element in vocab]
		tokens = list(set(vocab)) # Gets the unique words in the corpus
		token_size = len(tokens)

		word_to_index = {words: i for i, words in enumerate(tokens)}
		index_to_word = {index: word for word, index in word_to_index.items()}
		word_to_vec = {i: [np.random.uniform(-0.5, 0.5, size=self.vec_length), 
						   np.random.uniform(-0.5, 0.5, size=self.vec_length),
						   np.random.uniform(-0.5, 0.5, size=1),
						   np.random.uniform(-0.5, 0.5, size=1)]
						for i, _ in enumerate(tokens)}

		setattr(self, 'token_size', token_size)
		setattr(self, 'word_to_index', word_to_index)
		setattr(self, 'word_to_vec', word_to_vec)
		setattr(self, 'cache_matrix', np.zeros((2*token_size, 
												self.vec_length+1)))

		return vocab

	def cooccur_matrix(self):
		vocab = self.vocabulary()
		
		cooccurrences = sparse.coo_matrix((self.token_size+1, self.token_size+1))

		# Creates a symmetric window around each word
		win = 2 * self.window_size + 1
		l = list(map(lambda x: self.word_to_index[x], vocab))
		lpadded = win // 2 * [self.token_size] + l + win // 2 * [self.token_size] 

		out = [lpadded[i:(i + win)] for i in range(len(l))]
		assert len(out) == len(l)

		d = []

		# Harmonically decaying weights
		for r in range(self.window_size):
			d.append(1/(r+1))

		d = np.array(d[::-1] + d)

		# Creates the cooccurrence matrix
		for windows in out:
			r = [windows[self.window_size] for i in range(win-1)]
			c = windows[:self.window_size] + windows[self.window_size+1:]

			sparse_temp = sparse.coo_matrix((d, (r, c)), 
											shape=(self.token_size+1, 
												   self.token_size+1))
			cooccurrences += sparse_temp
			
		# Removes the padding row and column
		cooccurrences = sparse.coo_matrix(cooccurrences[:-1, :-1]) 

		return cooccurrences

	def train(self, epochs=4, x_max=3, alpha=0.75, learning_rate=0.0001, 
			  decay_rate=0.9, annealing_rate=0.0, eps=0.00001):		

		coocur_matrix= self.cooccur_matrix()

		values = coocur_matrix.data
		main_row = coocur_matrix.row
		context_row = coocur_matrix.col
		coocur_size = coocur_matrix.shape[0]
		
		del(coocur_matrix)

		for i in range(epochs):
			# Feedforward Pass
			vector_sums = np.zeros_like(getattr(self, 'cache_matrix'))
			total_cost=0

			for _, coord in enumerate(zip(main_row, context_row, values)):
				weight = (coord[-1]/x_max)**alpha if coord[-1] < x_max else 1
				main = coord[0]
				context = coord[1]

				inner_cost = np.dot(self.word_to_vec[main][0], 
									self.word_to_vec[context][1])\
							+ self.word_to_vec[main][2]\
							+ self.word_to_vec[context][3]\
							- np.log(coord[-1])

				total_cost += 0.5 * weight * inner_cost**2

				# Fills the vector_sums matrix with the sums in relation 
				# to the main and context word. `vector_sums` will then 
				# be used in the gradient to compute SGD. 
				vector_sums[main, :-1] += weight * inner_cost \
										* self.word_to_vec[context][1]
				vector_sums[context+self.token_size, :-1] += weight * inner_cost *\
												   		self.word_to_vec[main][0]
				vector_sums[main, -1] += weight * inner_cost \
									   * self.word_to_vec[context][3]				   		
				vector_sums[context+self.token_size, -1] += weight * inner_cost\
													 * self.word_to_vec[main][2]

			eta = np.exp(-annealing_rate * i)

			# Backward Pass
			for i in range(self.token_size):
				# Main vector cache
				self.cache_matrix[i, :-1] = decay_rate\
										  * self.cache_matrix[i, :-1]\
										  + (1-decay_rate)\
										  * vector_sums[i, :-1]**2
				# Context vector cache
				self.cache_matrix[i+self.token_size, :-1] = decay_rate \
											  * self.cache_matrix[i+self.token_size, :-1] \
										      + (1-decay_rate)\
										      * vector_sums[i+self.token_size, :-1]**2
				# Main bias cache
				self.cache_matrix[i, -1] = decay_rate \
										 * self.cache_matrix[i, -1] \
							  			 + (1-decay_rate) \
							  			 * vector_sums[i, -1]**2

				# Context bias cache
				self.cache_matrix[i+self.token_size, -1] = decay_rate\
												* self.cache_matrix[i+self.token_size, -1] \
									   			+ (1-decay_rate)\
									   			* vector_sums[i+self.token_size, -1]**2

				# Updates the word vectors and biases
				# Main word vector
				self.word_to_vec[i][0] =- eta * learning_rate\
										* vector_sums[i, :-1]\
								  		/ np.sqrt(self.cache_matrix[i, :-1]+eps)
				# Context word vector
				self.word_to_vec[i][1] =- eta * learning_rate\
										* vector_sums[i+self.token_size, :-1]\
								 		/ np.sqrt(self.cache_matrix[i+self.token_size, :-1]+eps)
				# Main word bias
				self.word_to_vec[i][2] =- eta * learning_rate\
										* vector_sums[i, -1]\
										/ np.sqrt(self.cache_matrix[i, -1]+eps)
				# Context word bias
				self.word_to_vec[i][3] =- eta * learning_rate\
										* vector_sums[i+self.token_size, -1]\
								  		/ np.sqrt(self.cache_matrix[i+self.token_size, -1]+eps)

	def save(self, filename_labels, filename_vectors):
		"""
		Saves the word labels to the file ``filename_labels`` and the
		word vectors (main, context, main-bias, and context-bias) to 
		``filename_vectors``.
		"""

		labels = self.word_to_index
		vectors = self.word_to_vec
		f_lab = open(filename_labels, "wb")
		f_vec = open(filename_vectors, 'wb')
		pickle.dump(labels, f_lab)
		pickle.dump(vectors, f_vec)
		f_lab.close()
		f_vec.close()

def main():
	import matplotlib.pyplot as plt

	file = 'C:\\Users\\lukas\\Documents\\NLP\\Word_Embedding\\test.txt'
	text = open(file, 'r').read()

	temp = GloVe(text)
	temp.train()

	word_vec = temp.word_to_vec
	points = np.zeros((len(word_vec), 2))

	for i, _ in enumerate(word_vec):
		points[i] = np.mean([word_vec[i][0], word_vec[i][1]], axis=0)

	word_to_index = temp.word_to_index
	labels = [word for word, _ in word_to_index.items()]

	plt.figure(figsize=(30, 20))

	for label, x, y, in zip(labels, points[:, 0], points[:, 1]):
		plt.annotate(label, xy=(x, y), xytext=(-0, 0),
	        		 textcoords='offset points', ha='right', va='bottom')

	plt.plot(*zip(*points), marker='o', color='r', ls='')
	plt.show()

if __name__ == '__main__':
	# pass
	main()
