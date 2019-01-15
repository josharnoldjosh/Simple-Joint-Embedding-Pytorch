from settings import config
import torch
import torch.nn as nn
import torch.nn.init

class TimeDistributed(nn.Module):
    """
    Time Distributed Wrapper Layer similar to keras

    Apply a module across each time step
    """
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class SentenceEmbedding(nn.Module):

    def __init__(self):
        super(SentenceEmbedding, self).__init__()

        self.time_distributed = TimeDistributed(module = nn.Linear(config['sentence_embedding_size'],
            config["linear_hidden"]), batch_first = True)

        self.rnn = nn.LSTM(config["linear_hidden"], config['lstm_hidden_size'], config["lstm_depth"], batch_first = True)

        self.encoder = nn.Linear(config['lstm_hidden_size'], config["joint_embedding_latent_space_dimension"])

        if torch.cuda.is_available():
            self.time_distributed.cuda()
            self.rnn.cuda()
            self.encoder.cuda()

        self.init_model_weights()

    def init_model_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, qa_pairs):       
        qa_pairs = qa_pairs.float()

        qa_pairs_emb = self.time_distributed(qa_pairs)
        
        _, (h_qa_t, _) = self.rnn(qa_pairs_emb)

        qa_hist_state = h_qa_t[-1]        
        
        diag_state = self.encoder(qa_hist_state)
        out = self.l2norm(diag_state)

        return out        


    def l2norm(self, X):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X    