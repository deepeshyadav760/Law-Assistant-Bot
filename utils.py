# utils.py
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import json
import pandas as pd

# Text preprocessing function
def preprocess_text(text):
    """Clean and tokenize text."""
    text = text.lower()
    text = re.sub(r'\d+', ' NUM ', text)  # Replace numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Create vocabulary function
def create_vocabulary(texts, min_freq=2):
    """Create vocabulary from texts."""
    word_freq = {}
    for text in texts:
        for word in text.split():
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    vocab = {
        '<PAD>': 0,
        '< SOS >': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    
    idx = 4
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

# Convert text to indices using vocabulary
def text_to_indices(text, vocab):
    """Convert text to indices using vocabulary."""
    tokens = text.split()
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return indices

# Function to generate response from the model
def generate_response(model, input_text, input_vocab, output_vocab, device, max_length=100):
    """Generate a response from the trained model."""
    model.eval()
    
    # Preprocess input text
    processed_text = preprocess_text(input_text)
    input_indices = [input_vocab['< SOS >']] + text_to_indices(processed_text, input_vocab) + [input_vocab['<EOS>']]
    src_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
    
    # Create inverse vocabulary for output
    idx_to_word = {idx: word for word, idx in output_vocab.items()}
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Start with < SOS > token
        input = torch.tensor([output_vocab['< SOS >']]).to(device)
        
        output_tokens = []
        
        for i in range(max_length):
            output, hidden, cell = model.decoder(input, hidden, cell)
            
            # Get the word with highest probability
            top_token = output.argmax(1).item()
            
            if top_token == output_vocab['<EOS>']:
                break
            
            if top_token != output_vocab['<PAD>'] and top_token != output_vocab['< SOS >']:
                output_tokens.append(idx_to_word.get(top_token, '<UNK>'))
            
            # Next input is the predicted token
            input = torch.tensor([top_token]).to(device)
    
    return ' '.join(output_tokens)

# Function to save model and vocabularies
def save_model(model, input_vocab, output_vocab, model_path='models/ipc_model.pt', 
               input_vocab_path='models/input_vocab.pkl', output_vocab_path='models/output_vocab.pkl'):
    """Save model and vocabularies to files."""
    try:
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Save vocabularies
        with open(input_vocab_path, 'wb') as f:
            pickle.dump(input_vocab, f)
        
        with open(output_vocab_path, 'wb') as f:
            pickle.dump(output_vocab, f)
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

# Function to load data from JSON file
def load_data_from_json(file_path):
    """Load IPC data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to calculate similarity score for keyword search
def calculate_similarity(user_input, crime_description):
    """Calculate similarity score between user input and crime description."""
    user_tokens = set(user_input.lower().split())
    crime_tokens = set(crime_description.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(user_tokens.intersection(crime_tokens))
    union = len(user_tokens.union(crime_tokens))
    
    return intersection / union if union > 0 else 0

# Function to find matching IPC sections using keyword search
def find_matching_sections(user_input, df, top_n=3):
    """Find matching IPC sections based on keyword similarity."""
    matches = []
    
    for _, row in df.iterrows():
        similarity = calculate_similarity(user_input, row['crime_description'])
        if similarity > 0:
            matches.append((similarity, row))
    
    # Sort by similarity score in descending order
    matches.sort(reverse=True, key=lambda x: x[0])
    
    return matches[:top_n] if matches else []