import torch
import torch.nn as nn
import torch.nn.functional as F

class RecogNet_gru(nn.Module):
    def __init__(self, embedding_dim):
        super(RecogNet_gru, self).__init__()
        self.hidden_dim = 512
        self.sent_rnn = nn.GRU( embedding_dim,
                                self.hidden_dim,
                                bidirectional=True,
                                batch_first=True)
        self.l1 = nn.Linear(self.hidden_dim, 4)

    def forward(self, x):
        b,s,w,e = x.shape
        x = x.view(b,s*w,e)
        x, __ = self.sent_rnn(x)
        x = x.view(b,s,w,-1)
        x = torch.max(x,dim=2)[0]
        x = x[:,:,:self.hidden_dim] + x[:,:,self.hidden_dim:]
        x = torch.max(x,dim=1)[0]
        x = torch.sigmoid(self.l1(F.relu(x)))
        return x






class RecogNet_lstm(nn.Module):
    def __init__(self, embedding_dim):
        super(RecogNet_lstm, self).__init__()

        self.embedding_dim=embedding_dim
        
        self.hidden_dim = 512
        
        self.rnn_lstm =nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True) 
        self.decoder = nn.Linear(self.hidden_dim, 4)

   

    def forward(self, inputs):
        b,s,w,e = inputs.shape
        inputs = inputs.view(b,s*w,e)
        states, hidden = self.rnn_lstm(inputs)
        
        x = states.view(b,s,w,-1)
        
        x = torch.max(x,dim=2)[0]
        x = x[:,:,:self.hidden_dim] + x[:,:,self.hidden_dim:]
        x = torch.max(x,dim=1)[0]

        outputs = torch.sigmoid(self.decoder(F.relu(x)))
        return outputs


