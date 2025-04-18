# app.py
import streamlit as st
import pandas as pd
import json
import os
import re
import random

# Set page configuration
st.set_page_config(
    page_title="IPC Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import optional dependencies with fallbacks
try:
    import torch
    from model import Encoder, Decoder, Seq2Seq
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.sidebar.warning("PyTorch not installed. Model-based predictions will not be available. Install with: pip install torch")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
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
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.sidebar.warning("NLTK not installed. Some text processing features will be limited. Install with: pip install nltk")

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False
    st.sidebar.warning("Pickle not available. Model loading will not work.")

# Import utils functions with fallbacks
if TORCH_AVAILABLE and NLTK_AVAILABLE:
    try:
        from utils import preprocess_text, text_to_indices, generate_response
        UTILS_AVAILABLE = True
    except ImportError:
        UTILS_AVAILABLE = False
        st.sidebar.warning("Utils module not found. Make sure utils.py is in the same directory.")
else:
    UTILS_AVAILABLE = False

# Simplified text preprocessing when NLTK is not available
def simple_preprocess(text):
    """Basic text preprocessing without NLTK."""
    text = text.lower()
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

# Simple similarity function for keyword matching
def calculate_similarity(text1, text2):
    """Calculate simple word overlap similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    return len(intersection) / max(len(words1), 1)

# Load data
@st.cache_data
def load_data():
    data_paths = [
        'data/ipc_data.json',
        'ipc_data.json',
        'data/ipc_law_assistant_1000_entries.json',
        'ipc_law_assistant_1000_entries.json'
    ]
    
    for path in data_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except FileNotFoundError:
            continue
    
    return None

# Load model and vocabularies if PyTorch is available
@st.cache_resource
def load_model():
    if not all([TORCH_AVAILABLE, PICKLE_AVAILABLE, UTILS_AVAILABLE]):
        return None, None, None
    
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
                hidden_size = 512  # Match the size used during training
                
                encoder = Encoder(input_dim, hidden_size, num_layers=2, dropout=0.2)
                decoder = Decoder(output_dim, hidden_size, num_layers=2, dropout=0.2)
                
                model = Seq2Seq(encoder, decoder, device).to(device)
                
                # Load trained parameters
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                return model, input_vocab, output_vocab
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")
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
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    except:
        st.sidebar.warning("Could not create directories. This might be expected in cloud environments.")
    
    # Status information
    st.sidebar.subheader("System Status")
    if TORCH_AVAILABLE:
        st.sidebar.success("✅ PyTorch available")
    else:
        st.sidebar.error("❌ PyTorch not available")
        
    if NLTK_AVAILABLE:
        st.sidebar.success("✅ NLTK available")
    else:
        st.sidebar.error("❌ NLTK not available")
    
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
                try:
                    with open('data/ipc_data.json', 'w') as f:
                        json.dump(data, f)
                    st.sidebar.success("Data uploaded and saved!")
                except:
                    st.sidebar.warning("Data loaded but couldn't save to disk. App will work for this session only.")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
    else:
        st.sidebar.success(f"Loaded {len(df)} IPC sections")
    
    # Load model if PyTorch is available
    model, input_vocab, output_vocab = (None, None, None)
    if TORCH_AVAILABLE:
        model, input_vocab, output_vocab = load_model()
        model_status = "Model loaded" if model is not None else "Model not found"
        st.sidebar.text(f"Model Status: {model_status}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Chat with Assistant", "Browse IPC Data", "Model Information"])
    
    # Chat input - MUST BE OUTSIDE TABS
    user_input = st.chat_input("Describe a crime scenario or ask about an IPC section...")
    
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
        
        # Process user input
        if user_input:
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
                        elif all([TORCH_AVAILABLE, UTILS_AVAILABLE, model is not None, input_vocab is not None, output_vocab is not None]):
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
                                    similarity = calculate_similarity(user_input_lower, crime_desc)
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
                                similarity = calculate_similarity(user_input_lower, crime_desc)
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
        
        # Model testing section (only if PyTorch is available)
        if TORCH_AVAILABLE and model is not None:
            st.subheader("Test Model")
            
            test_input = st.text_area("Enter a crime description to test:", "Someone broke into a house at night and stole valuable items.")
            
            if st.button("Generate Prediction"):
                st.write("**Model Prediction**:")
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                prediction = generate_response(model, test_input, input_vocab, output_vocab, device)
                st.write(prediction)
        else:
            st.warning("Model testing is not available. Either PyTorch is not installed or the model wasn't loaded successfully.")
            
            if df is not None:
                st.subheader("Keyword Matching (Fallback Method)")
                test_input = st.text_area("Enter a crime description to test:", "Someone broke into a house at night and stole valuable items.")
                
                if st.button("Find Matching Sections"):
                    # Use keyword matching
                    matches = []
                    for _, row in df.iterrows():
                        crime_desc = row['crime_description'].lower()
                        similarity = calculate_similarity(test_input.lower(), crime_desc)
                        if similarity > 0:
                            matches.append((similarity, row))
                    
                    if matches:
                        matches.sort(reverse=True, key=lambda x: x[0])
                        best_match = matches[0][1]
                        
                        st.markdown(f"**Best Match**: Section {best_match['ipc_section']} - {best_match['ipc_title']}")
                        st.markdown(f"**Similarity Score**: {matches[0][0]:.2f}")
                        
                        if len(matches) > 1:
                            st.markdown("**Other Potential Matches**:")
                            for i, (score, row) in enumerate(matches[1:5]):  # Show up to 5 more matches
                                st.markdown(f"- Section {row['ipc_section']}: {row['ipc_title']} (Score: {score:.2f})")

if __name__ == "__main__":
    main()
