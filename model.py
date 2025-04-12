# model.py
import torch
import torch.nn as nn
import random

# Define the LSTM Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, hidden, cell

# Define the LSTM Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        # x shape: (batch_size, 1)
        x = x.unsqueeze(1)  # (batch_size, 1)
        
        embedded = self.embedding(x)  # (batch_size, 1, hidden_size)
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        prediction = self.fc(output.squeeze(1))  # (batch_size, output_size)
        
        return prediction, hidden, cell

# Define the Sequence-to-Sequence model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source shape: (batch_size, source_len)
        # target shape: (batch_size, target_len)
        
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Get encoder outputs
        encoder_outputs, hidden, cell = self.encoder(source)
        
        # First input to the decoder is the < SOS > token
        input = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            outputs[:, t] = output
            
            # Teacher forcing: use ground truth as next input or use best prediction
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            input = target[:, t] if teacher_force else top1
            
        return outputs

        
        prediction = self.fc(output.squeeze(1))  # (batch_size, output_size)
        
        return prediction, hidden, cell

# Define the Sequence-to-Sequence model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source shape: (batch_size, source_len)
        # target shape: (batch_size, target_len)
        
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Get encoder outputs
        encoder_outputs, hidden, cell = self.encoder(source)
        
        # First input to the decoder is the < SOS > token
        input = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            outputs[:, t] = output
            
            # Teacher forcing: use ground truth as next input or use best prediction
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            input = target[:, t] if teacher_force else top1
            
        return outputs