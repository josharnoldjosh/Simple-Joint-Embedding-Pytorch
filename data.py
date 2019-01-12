from settings import config
import deepdish.io as dd
import numpy
import pickle
from j.osh import *
import torch
from torch.autograd import Variable

class Loader(object):
	"""
	Data Loader for the joint embedding.
	"""

	def __init__(self):		
		"""
		Initialize the data loader with default settings.
		"""

		# initialize variables
		self.data = ([], [])
		self.batch_size = config["batch_size"]
		self.batch_number = 0
		self.word_to_index = {'<blank>':0, '<sos>':1, '<eos>':2, '<unk>':3}
		self.index_to_word = {0:'<blank>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}

		# load dataset
		self.load_dataset()
		
	def load_dataset(self):
		"""
		Loads a dataset from file into the class.
		"""

		# load captions from file
		captions = dd.load("data/"+config["dataset"]+"/"+config["dataset"]+"_caps.h5")

		# load image features from file 
		image_features = numpy.load("data/"+config["dataset"]+"/"+config["dataset"]+'_ims.npy')

		# update data
		self.data = (captions, image_features)

		if len(captions) != len(image_features):
			print("[WARNING] The number of captions do not match the number of image feature vectors.")

		print("[LOADED] Dataset successfully loaded: ", config["dataset"])

	def create_dictionaries(self):
		"""
		Creates/overwrites dictionaries for embedding literal strings to vectors and back.
		"""
		captions = self.data[0]
		offset = len(self.word_to_index)

		# add each word in each caption to a set
		words = set()
		for idx, caption in enumerate([item for caption_set in captions for item in caption_set]):
			for word in caption.split():
				words.add(word)

		# assigns each word to an index to relate words to numbers and vice versa.
		for idx, word in enumerate(words):
			self.word_to_index[word] = idx + offset
			self.index_to_word[idx+offset] = word

		print("[INIT] Dictionaries created.")

	def save_dictionaries(self):
		"""
		Saves the current dictionaries to file for later recall.
		"""
		ensure_dir("dict/") # ensure the dict/ folder exists

		with open('dict/word_to_index.pkl', 'wb') as file:
			pickle.dump(self.word_to_index, file)
		with open('dict/index_to_word.pkl', 'wb') as file:
			pickle.dump(self.index_to_word, file)

		print("[SAVED] Dictionaries \"word_to_index.pkl\" and \"index_to_word.pkl\" saved to dict/")

	def load_dictionaries(self):
		"""
		Loads embedded dictionaries from file into the data loader class.
		"""
		try:
			self.word_to_index = pickle.load(open('dict/word_to_index.pkl', 'rb'))
			self.index_to_word = pickle.load(open('dict/index_to_word.pkl', 'rb'))
		except:
			print("[ERROR] Unable to load dictionaries: \"word_to_index.pkl\" and \"index_to_word.pkl\".")
		print("[LOADED] Dictionaries loaded: \"word_to_index.pkl\" and \"index_to_word.pkl\".")

	def __iter__(self):
		return self

	def __next__(self):
		"""
		Returns the next batch in the data loader iterator.
		"""

		# define upper and lower index bounds
		upper_idx = (self.batch_number+1)*self.batch_size
		lower_idx = self.batch_number*self.batch_size

		if lower_idx < len(self.data[0]):
			# increment batch number
			self.batch_number += 1

			# extract captions and image features
			captions = self.data[0][lower_idx:upper_idx]
			image_features = self.data[1][lower_idx:upper_idx]

			# prepare our data to go into the model
			captions, image_features = self.preprocess(captions, image_features)
			return captions, image_features

		self.batch_number = 0
		raise StopIteration

	def preprocess(self, captions, image_features):
		"""
		Prepares the captions and image features to go directly into the model.
		"""
		def get_seq(caption): 
			seq = [self.word_to_index[word] if word in self.word_to_index.keys() else 1 for word in caption.split()]       	
			seq.insert(0, self.word_to_index["<sos>"])
			seq.append(self.word_to_index["<eos>"])
			if len(seq) > config["sentence_embedding_size"]:
				print("[WARNING] sentence_embedding_size size exceeded. Check your settings.")			
			return seq

		# Convert a caption to an array of indexes of words from the dictionary, self.word_to_index  
		sequences = []
		for idx, caption in enumerate(captions):
			sequences.append([get_seq(i) for i in caption])


		# Create a 3D matrix
		x = len(sequences)
		y = max([len(i) for i in sequences])
		y = max([len(j) for i in sequences for j in i])
		z = config["sentence_embedding_size"]

		processed_captions = numpy.zeros((x, y, z)).astype('int64')		

		# for an array of seqs
		for idx, seq in enumerate(sequences):
			# for a seq array
			for jdx, item in enumerate(seq):
				# for each element in the seq array
				for kdx, element in enumerate(item):
					# set value in matrix
					processed_captions[idx][jdx][kdx] = element

		# Just convert image features to numpy array
		processed_image_features = numpy.asarray(image_features, dtype=numpy.float32)

		if torch.cuda.is_available():
			return Variable(torch.from_numpy(processed_captions)).cuda(), Variable(torch.from_numpy(processed_image_features)).cuda()

		return Variable(torch.from_numpy(processed_captions)), Variable(torch.from_numpy(processed_image_features))