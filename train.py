# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pickle
import json
import pandas as pd
import os
import argparse
import time
import random

from model import Encoder, Decoder, Seq2Seq
from utils import preprocess_text, create_vocabulary, text_to_indices, save_model

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

download_nltk_resources()

# Define the dataset class
class IPCDataset(Dataset):
    def __init__(self, crime_descriptions, ipc_info, vocab_input, vocab_output):
        self.crime_descriptions = crime_descriptions
        self.ipc_info = ipc_info
        self.vocab_input = vocab_input
        self.vocab_output = vocab_output
        
    def __len__(self):
        return len(self.crime_descriptions)
    
    def __getitem__(self, idx):
        input_text = self.crime_descriptions[idx]
        output_text = self.ipc_info[idx]
        
        input_indices = [self.vocab_input['< SOS >']] + text_to_indices(input_text, self.vocab_input) + [self.vocab_input['<EOS>']]
        output_indices = [self.vocab_output['< SOS >']] + text_to_indices(output_text, self.vocab_output) + [self.vocab_output['<EOS>']]
        
        return {
            'input': torch.tensor(input_indices, dtype=torch.long),
            'output': torch.tensor(output_indices, dtype=torch.long)
        }

# Define collate function for dataloader
def collate_fn(batch):
    input_sequences = [item['input'] for item in batch]
    output_sequences = [item['output'] for item in batch]
    
    input_sequences_padded = pad_sequence(input_sequences, batch_first=True, padding_value=0)
    output_sequences_padded = pad_sequence(output_sequences, batch_first=True, padding_value=0)
    
    return {
        'input': input_sequences_padded,
        'output': output_sequences_padded
    }

# Training function
def train_model(model, data_loader, optimizer, criterion, device, clip=1):
    model.train()
    epoch_loss = 0
    
    for batch in data_loader:
        src = batch['input'].to(device)
        trg = batch['output'].to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # Reshape for CrossEntropyLoss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Skip < SOS > token
        trg = trg[:, 1:].reshape(-1)  # Skip < SOS > token
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

# Model evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            src = batch['input'].to(device)
            trg = batch['output'].to(device)
            
            output = model(src, trg, 0)  # No teacher forcing
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

# Function to train the model with the data
def train_ipc_model(data_json_path, hidden_size=256, n_layers=2, dropout=0.2, 
                   batch_size=32, learning_rate=0.001, epochs=20, save_dir='models'):
    """Train the IPC law assistant model with the provided data."""
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model save paths
    model_path = os.path.join(save_dir, 'ipc_model.pt')
    input_vocab_path = os.path.join(save_dir, 'input_vocab.pkl')
    output_vocab_path = os.path.join(save_dir, 'output_vocab.pkl')
    
    # Load data
    try:
        with open(data_json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    print(f"Loaded {len(df)} IPC entries for training")
    
    # Preprocess data
    crime_descriptions = [preprocess_text(desc) for desc in df['crime_description']]
    
    # Combine relevant columns for output
    ipc_info = [
        preprocess_text(f"section {row['ipc_section']} {row['ipc_title']} {row['ipc_description']} case {row['example_case']} verdict {row['verdict_summary']}")
        for _, row in df.iterrows()
    ]
    
    # Create vocabularies
    input_vocab = create_vocabulary(crime_descriptions)
    output_vocab = create_vocabulary(ipc_info)
    
    print(f"Input vocabulary size: {len(input_vocab)}")
    print(f"Output vocabulary size: {len(output_vocab)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        crime_descriptions, ipc_info, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = IPCDataset(X_train, y_train, input_vocab, output_vocab)
    test_dataset = IPCDataset(X_test, y_test, input_vocab, output_vocab)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)
    
    encoder = Encoder(input_dim, hidden_size, num_layers=n_layers, dropout=dropout)
    decoder = Decoder(output_dim, hidden_size, num_layers=n_layers, dropout=dropout)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    
    # Training loop
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss = train_model(model, train_dataloader, optimizer, criterion, device)
        valid_loss = evaluate_model(model, test_dataloader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1}/{epochs} | Time: {epoch_mins}m {epoch_secs:.0f}s')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {valid_loss:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save the model and vocabularies
            save_model(model, input_vocab, output_vocab, model_path, input_vocab_path, output_vocab_path)
            print(f"Model saved (Epoch {epoch+1}) - Valid Loss: {valid_loss:.4f}")
    
    print("Training completed!")
    return model, input_vocab, output_vocab

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IPC Law Assistant Model')
    parser.add_argument('--data', type=str, required=True, help='Path to IPC JSON data file')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save model files')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of LSTM')
    parser.add_argument('--layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("Starting IPC Law Assistant Model Training...")
    model, input_vocab, output_vocab = train_ipc_model(
        args.data, 
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    print("Training completed successfully!")