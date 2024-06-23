
import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self,  vocab_size, args, name):
        """
        """
        super(RNN, self).__init__()

        self.name = name
        self.Len = 0
        self.loss = 0.0
        self.vocab_size = vocab_size 
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=0.5)  
        self.fc = nn.Linear(self.hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        self.rnn.flatten_parameters()
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


class RNN50(nn.Module):
    def __init__(self,  vocab_size, args, name):

        super(RNN50, self).__init__()

        self.name = name
        self.Len = 0
        self.loss = 0.0
        self.vocab_size = vocab_size  
        self.embedding_dim = 50
        self.hidden_dim = 256
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=0.5)  
        self.fc = nn.Linear(self.hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        self.rnn.flatten_parameters()
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out