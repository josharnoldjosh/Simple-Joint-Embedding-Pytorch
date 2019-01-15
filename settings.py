config = {

	# Model input
	"dataset":"visual_dialog", # the path that points to the data folder
	"image_dimension":512, # the size of the image feature vectors loaded as an input, e.g, 512 for Resnet

	# training parameters
	"num_epochs":100, # number of epochs to train on
	"batch_size":64, # the number of elements per batch	
	"learning_rate":0.00001, # learning rate for gradient descent
	"grad_clip":2.0, # clip the gradient during gradient descent
	"display_freq":20, # display the loss every x amount of times

	# model parameters
	"sentence_embedding_size":64, # the maximum number of words in a sentence to be embedded into a vector
	"lstm_hidden_size":1000, # the output vector size of the LSTM hidden layer
	"joint_embedding_latent_space_dimension":1000, # the dimension size of the latent space of our embedding
	"margin_pairwise_ranking_loss":0.2, # factor for the pairwise ranking loss function

	# time distributed
	"linear_hidden":1000,
	"lstm_depth":1
}	
