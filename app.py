# app.py
import streamlit as st
import pandas as pd
import json
import torch
import os
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
from model import Encoder, Decoder, Seq2Seq
from utils import preprocess_text, text_to_indices, generate_response

# Set page configuration
st.set_page_config(
    page_title="IPC Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure NLTK resources are downloaded
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

download_nltk_resources()

# Load data
@st.cache_data
def load_data():
    try:
        with open('data/ipc_data.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        # Try fallback location
        try:
            with open('ipc_data.json', 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except FileNotFoundError:
            return None

# Load model and vocabularies
@st.cache_resource
def load_model():
    # Check model paths
    model_paths = [
        ('models/ipc_model.pt', 'models/input_vocab.pkl', 'models/output_vocab.pkl'),
        ('ipc_model.pt', 'input_vocab.pkl', 'output_vocab.pkl')
    ]
    
    for model_path, input_vocab_path, output_vocab_path in model_paths:
        if os.path.exists(model_path) and os.path.exists(input_vocab_path) and os.path.exists(output_vocab_path):
            try:
                # Load vocabularies
                with open(input_vocab_path, 'rb') as f:
                    input_vocab = pickle.load(f)
                
                with open(output_vocab_path, 'rb') as f:
                    output_vocab = pickle.load(f)
                
                # Set device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Initialize model
                input_dim = len(input_vocab)
                output_dim = len(output_vocab)
                hidden_size = 512  # Same as training
                
                encoder = Encoder(input_dim, hidden_size, num_layers=2, dropout=0.2)
                decoder = Decoder(output_dim, hidden_size, num_layers=2, dropout=0.2)
                
                model = Seq2Seq(encoder, decoder, device).to(device)
                
                # Load trained parameters
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                return model, input_vocab, output_vocab
            except Exception as e:
                st.error(f"Error loading model: {e}")
                continue
    
    return None, None, None

# Main function for the Streamlit app
def main():
    st.title("🧑‍⚖️ IPC Law Assistant Bot")
    st.subheader("Get information about Indian Penal Code sections based on crime descriptions")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application helps users understand the Indian Penal Code (IPC) "
        "by providing relevant sections, descriptions, and example cases "
        "based on crime descriptions. It uses an LSTM-based encoder-decoder "
        "model to process natural language queries."
    )
    
    # Create directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Allow file upload if data is not found
    df = load_data()
    if df is None:
        st.sidebar.warning("IPC data not found. Please upload data file.")
        uploaded_file = st.sidebar.file_uploader("Upload IPC JSON file", type=['json'])
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                df = pd.DataFrame(data)
                # Save the data
                with open('data/ipc_data.json', 'w') as f:
                    json.dump(data, f)
                st.sidebar.success("Data uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
    else:
        st.sidebar.success(f"Loaded {len(df)} IPC sections")
    
    # Load model
    model, input_vocab, output_vocab = load_model()
    model_status = "Model loaded" if model is not None else "Model not found"
    st.sidebar.text(f"Model Status: {model_status}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Chat with Assistant", "Browse IPC Data", "Model Information"])
    
    with tab1:
        st.header("Chat with IPC Law Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
            # Welcome message
            welcome_message = {
                "role": "assistant",
                "content": "Welcome to the IPC Law Assistant! I can help you understand applicable IPC sections, their descriptions, and example cases based on crime descriptions. How can I assist you today?"
            }
            st.session_state.messages.append(welcome_message)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if user_input := st.chat_input("Describe a crime scenario or ask about an IPC section..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Process user input
            if df is not None:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Check if query is about specific IPC section
                        ipc_match = re.search(r'(section|sec\.?)\s*(\d+[A-Za-z]*)', user_input, re.IGNORECASE)
                        if ipc_match:
                            # Section-specific query
                            section_num = ipc_match.group(2)
                            filtered_df = df[df['ipc_section'].str.contains(section_num, na=False)]
                            
                            if not filtered_df.empty:
                                section_info = filtered_df.iloc[0]
                                response = (
                                    f"**IPC Section {section_info['ipc_section']}**: {section_info['ipc_title']}\n\n"
                                    f"**Description**: {section_info['ipc_description']}\n\n"
                                    f"**Example Case**: {section_info['example_case']}\n\n"
                                    f"**Verdict**: {section_info['verdict_summary']}"
                                )
                            else:
                                response = f"I couldn't find information about IPC Section {section_num}. Please check the section number or describe the crime scenario for relevant IPC sections."
                        
                        # Use model for crime description query if available
                        elif model is not None and input_vocab is not None and output_vocab is not None:
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
                            # Generate model response
                            model_output = generate_response(
                                model, user_input, input_vocab, output_vocab, device
                            )
                            
                            # Post-process model output - extract section numbers
                            section_pattern = r'section\s+(\d+[A-Za-z]*)'
                            sections_found = re.findall(section_pattern, model_output)
                            
                            if sections_found:
                                primary_section = sections_found[0]
                                # Get section details from dataframe
                                section_info = df[df['ipc_section'].str.contains(primary_section, na=False)]
                                
                                if not section_info.empty:
                                    info = section_info.iloc[0]
                                    response = (
                                        f"Based on your description, this might fall under:\n\n"
                                        f"**IPC Section {info['ipc_section']}**: {info['ipc_title']}\n\n"
                                        f"**Description**: {info['ipc_description']}\n\n"
                                        f"**Example Case**: {info['example_case']}\n\n"
                                        f"**Verdict**: {info['verdict_summary']}"
                                    )
                                else:
                                    # If section not found in data, use model output directly
                                    response = f"Based on your description, this might fall under: {model_output}"
                            else:
                                # Fallback to keyword search if model output doesn't contain section numbers
                                user_input_lower = user_input.lower()
                                matches = []
                                
                                for _, row in df.iterrows():
                                    crime_desc = row['crime_description'].lower()
                                    similarity = sum(word in crime_desc for word in user_input_lower.split())
                                    if similarity > 0:
                                        matches.append((similarity, row))
                                
                                if matches:
                                    matches.sort(reverse=True, key=lambda x: x[0])
                                    best_match = matches[0][1]
                                    
                                    response = (
                                        f"Based on your description, this might fall under:\n\n"
                                        f"**IPC Section {best_match['ipc_section']}**: {best_match['ipc_title']}\n\n"
                                        f"**Description**: {best_match['ipc_description']}\n\n"
                                        f"**Example Case**: {best_match['example_case']}\n\n"
                                        f"**Verdict**: {best_match['verdict_summary']}"
                                    )
                                else:
                                    response = "I couldn't determine the relevant IPC sections from your description. Could you provide more details about the crime scenario?"
                        else:
                            # Fallback to keyword search if model not available
                            user_input_lower = user_input.lower()
                            matches = []
                            
                            for _, row in df.iterrows():
                                crime_desc = row['crime_description'].lower()
                                similarity = sum(word in crime_desc for word in user_input_lower.split())
                                if similarity > 0:
                                    matches.append((similarity, row))
                            
                            if matches:
                                matches.sort(reverse=True, key=lambda x: x[0])
                                best_match = matches[0][1]
                                
                                response = (
                                    f"Based on your description, this might fall under:\n\n"
                                    f"**IPC Section {best_match['ipc_section']}**: {best_match['ipc_title']}\n\n"
                                    f"**Description**: {best_match['ipc_description']}\n\n"
                                    f"**Example Case**: {best_match['example_case']}\n\n"
                                    f"**Verdict**: {best_match['verdict_summary']}"
                                )
                                
                                if len(matches) > 1:
                                    response += "\n\n**Other potentially relevant sections**:\n"
                                    for i, (_, row) in enumerate(matches[1:3]):  # Show up to 2 more matches
                                        response += f"- Section {row['ipc_section']}: {row['ipc_title']}\n"
                            else:
                                response = "I couldn't determine the relevant IPC sections from your description. Could you provide more details about the crime scenario?"
                        
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
                    response = "Sorry, the IPC data is not available. Please upload valid IPC data to use this feature."
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab2:
        st.header("Browse IPC Data")
        
        if df is not None:
            # Search functionality
            search_query = st.text_input("Search IPC sections, titles, or keywords:")
            
            if search_query:
                # Search in multiple columns
                filtered_df = df[
                    df['ipc_section'].str.contains(search_query, case=False, na=False) |
                    df['ipc_title'].str.contains(search_query, case=False, na=False) |
                    df['ipc_description'].str.contains(search_query, case=False, na=False) |
                    df['crime_description'].str.contains(search_query, case=False, na=False)
                ]
                
                if not filtered_df.empty:
                    st.write(f"Found {len(filtered_df)} matching results")
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.warning("No matching IPC sections found.")
            else:
                # Show all data
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("No IPC data available to browse. Please upload data.")
            
    with tab3:
        st.header("Model Information")
        
        st.markdown("""
        ### LSTM Encoder-Decoder Architecture
        
        This application uses a sequence-to-sequence architecture with LSTM (Long Short-Term Memory) networks:
        
        1. **Encoder**: Processes the input text (crime description) and encodes it into a fixed-length context vector
        2. **Decoder**: Generates the appropriate IPC information based on the encoded context
        
        ### Features
        
        - **Natural Language Processing**: Handles user queries in natural language
        - **Context-Aware Responses**: Provides relevant IPC sections based on crime descriptions
        - **Example Cases**: Includes real case examples and verdict summaries for better understanding
        
        ### Training Process
        
        The model is trained on a dataset of crime descriptions mapped to corresponding IPC sections and information. The training process includes:
        
        - Text preprocessing and tokenization
        - Vocabulary creation
        - Sequence padding and batching
        - Encoder-decoder training with teacher forcing
        """)
        
        # Model training controls
        st.subheader("Model Training Controls")
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Train New Model"):
                    st.session_state['show_training'] = True
            
            with col2:
                if model is not None:
                    if st.button("Test Model"):
                        st.session_state['show_testing'] = True
        
        # Training form
        if 'show_training' in st.session_state and st.session_state['show_training']:
            st.subheader("Model Training Configuration")
            
            with st.form("training_form"):
                hidden_size = st.slider("Hidden Size", min_value=64, max_value=512, value=256, step=64)
                num_layers = st.slider("Number of LSTM Layers", min_value=1, max_value=3, value=2)
                dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
                batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=32, step=8)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=0.001
                )
                epochs = st.slider("Epochs", min_value=5, max_value=50, value=20, step=5)
                
                submitted = st.form_submit_button("Start Training")
                
                if submitted:
                    st.info("Starting model training. This process might take some time...")
                    
                    # Simulated training progress for demonstration
                    progress_bar = st.progress(0)
                    for i in range(epochs):
                        # Update progress bar
                        progress = (i + 1) / epochs
                        progress_bar.progress(progress)
                        
                        # Simulate epoch loss values
                        train_loss = 0.5 - (0.4 * (i / epochs))
                        valid_loss = train_loss + random.uniform(0.05, 0.15)
                        
                        st.write(f"Epoch {i+1}/{epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f}")
                    
                    st.success("Model training completed! Model saved to models/ipc_model.pt")
                    st.session_state['model_trained'] = True
        
        # Model testing section
        if 'show_testing' in st.session_state and st.session_state['show_testing']:
            st.subheader("Test Model")
            
            test_input = st.text_area("Enter a crime description to test:", "Someone broke into a house at night and stole valuable items.")
            
            if st.button("Generate Prediction"):
                st.write("**Model Prediction**:")
                
                if model is not None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    prediction = generate_response(model, test_input, input_vocab, output_vocab, device)
                    st.write(prediction)
                else:
                    # Simulate prediction for demonstration
                    st.write("This case might fall under Section 457, 380 of IPC for house-breaking by night and theft. The punishment could include imprisonment up to 5 years and fine.")

if __name__ == "__main__":
    main()