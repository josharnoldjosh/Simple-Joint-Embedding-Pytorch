import torch
import torch.nn.functional as F
from settings import config

from time_dist import SentenceEmbedding

class Model(torch.nn.Module):
	"""
	The model class for the joint embedding.
	"""

	def __init__(self):
		"""
		Initialize the model - setup all the layers and initialize weights.
		"""
		super(Model, self).__init__()

		# input output files
		self.input_name = "best.pkl"
		self.output_name = "best.pkl"

		# score
		self.score = 0

		# caption neural networks
		self.lstm = torch.nn.LSTM(config['sentence_embedding_size'], config['lstm_hidden_size'], 1, batch_first=True)
		self.caption_linear = torch.nn.Linear(config['lstm_hidden_size'], config["joint_embedding_latent_space_dimension"])

		self.sentence_pass = SentenceEmbedding()

		# image feature neural networks
		self.image_feature_linear = torch.nn.Linear(config['image_dimension'], config['joint_embedding_latent_space_dimension'])

		# initialize weights
		self.initialize_weights()		

		# optional - cuda
		if torch.cuda.is_available():			
			self.lstm.cuda()
			self.caption_linear.cuda()
			self.image_feature_linear.cuda()

		print("[INIT] Model successfully loaded.")

	def initialize_weights(self):
		"""
		Initializes the weights for the linear layers using xavier uniform.
		"""
		# captions
		torch.nn.init.xavier_uniform_(self.caption_linear.weight)
		self.caption_linear.bias.data.fill_(0)

		# image features
		torch.nn.init.xavier_uniform_(self.image_feature_linear.weight)
		self.image_feature_linear.bias.data.fill_(0)

	def forward(self, captions, image_features):
		"""
		Pass data through model
		"""
		return self.forward_caption(captions), self.forward_image(image_features)

	def forward_caption(self, captions):
		"""
		Pass captions through model.		
		"""
		return self.sentence_pass(captions)
		# sentence_embedding = captions.float()
		# _, (sentence_embedding, _) = self.lstm(sentence_embedding)		
		# x_sentence_embedding = sentence_embedding.squeeze(0)		
		# x_sentence_embedding = self.caption_linear(x_sentence_embedding)	
		# norm_x_sentence_embedding =  F.normalize(x_sentence_embedding, p=2, dim=1)	
		# return norm_x_sentence_embedding

	def forward_image(self, image):
		"""
		Pass image features through the model.
		"""		
		image_embedding = self.image_feature_linear(image)		
		norm_image_embedding = F.normalize(image_embedding, p=2, dim=1)
		return norm_image_embedding

	def save(self):
		"""
		Save the model to file.
		"""			
		torch.save(self.state_dict(), self.output_name+'.pkl')
		print("[SAVED] Model saved to file as", self.output_name)
		return

	def save_if_better(self, score):
		"""
		Save the model if the new score is better than current model score.
		Then update model score.
		"""
		if score > self.score:
			self.score = score
			self.save()			

	def load(self):
		"""
		Loads a models weights into the file.
		"""
		self.load_state_dict(torch.load(self.input_name+".pkl"))
		print("[LOADED] Model loaded as", self.input_name+".pkl", "\n")
		return	
