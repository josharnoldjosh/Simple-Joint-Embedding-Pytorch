import data as Data
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer
import evaluate
from settings import config

# load data
data = Data.Loader()
data.create_dictionaries()
data.save_dictionaries()

# load evaluation data
evaluation_data = Data.Loader()
evaluation_data.load_dictionaries()

# init model
model = Model()

# init loss
loss = Loss()

# init optimizer
optimizer = Optimizer(model)

def evaluate_model():
	"""
	Evaluate the model and print the Recall@K score.
	"""

	print("\n[RESULTS]")

	text_to_image, image_to_text = [], []

	# process in batches
	for captions, image_features in evaluation_data:
		
		# pass batch through model
		captions, image_features = model(captions, image_features)

		# evaluate text to image score
		text_to_image.append(evaluate.text_to_image(captions, image_features))
		image_to_text.append(evaluate.image_to_text(captions, image_features))

	# retreive score
	score = evaluate.recall_score(text_to_image, image_to_text)

	# save model if better
	model.save_if_better(score)

def train_model():
	"""
	Train the model with parameters from settings.py
	"""

	# train model
	for epoch in range(1, config["num_epochs"]+1):

		print("\n[EPOCH]", epoch)

		# process in batches
		for captions, image_features in data:

			# pass batch through model
			captions, image_features = model(captions, image_features)
			
			# cost
			cost = loss(captions, image_features)

			# backprop
			optimizer.backprop(cost)

		evaluate_model()

if __name__ == '__main__':
	train_model()
	evaluate_model()
	print("\n[DONE] Script complete.")