import torch
from settings import config

class Optimizer:
	"""
	The class for performing gradient descent on the model.
	"""

	def __init__(self, model):
		"""
		Initialize the optimizer with model parameters.
		"""
		self.learning_rate = config["learning_rate"]
		self.params = filter(lambda p: p.requires_grad, model.parameters())
		self.optimizer = torch.optim.Adam(self.params, self.learning_rate)
		self.display_freq = config["display_freq"]
		self.display_count = 0

	def backprop(self, cost):
		"""
		Will back propagate the cost through the model.
		"""

		# display loss
		if self.display_count % self.display_freq == 0 or self.display_count == 0:
			print("[LOSS] ", cost.data.cpu().numpy())

		self.display_count +=1

		# reset gradient
		self.optimizer.zero_grad()

		# back propagate
		cost.backward()

		# clip grad
		torch.nn.utils.clip_grad_norm_(self.params, config["grad_clip"])

		# step
		self.optimizer.step()
