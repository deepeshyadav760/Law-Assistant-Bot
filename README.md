# IPC Law Assistant Bot

A conversational AI assistant to help understand Indian Penal Code (IPC) sections based on crime descriptions. This application uses an LSTM-based encoder-decoder model to provide relevant IPC sections, descriptions, and example cases.

## Features

- **Natural Language Processing**: Chat with the assistant using natural language to describe crime scenarios
- **IPC Section Lookup**: Get information about specific IPC sections by number
- **Crime Description Analysis**: Receive relevant IPC sections based on crime descriptions
- **Example Cases**: View real-world case examples and verdicts for each IPC section
- **Interactive UI**: User-friendly interface with chat functionality and search options

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/ipc-law-assistant.git
cd ipc-law-assistant
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

4. Download NLTK resources (this will happen automatically on first run, or you can run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Project Structure

```
ipc_law_assistant/
│
├── data/
│   └── ipc_data.json                # JSON file containing IPC sections data
│
├── models/                          # Directory to store trained models
│   ├── ipc_model.pt                 # Trained PyTorch model
│   ├── input_vocab.pkl              # Pickled input vocabulary
│   └── output_vocab.pkl             # Pickled output vocabulary
│
├── app.py                           # Main Streamlit application
├── model.py                         # LSTM Encoder-Decoder model classes
├── train.py                         # Script for model training and data preprocessing
├── utils.py                         # Utility functions for text processing
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## Usage

### Running the Application

Start the Streamlit application:
```
streamlit run app.py
```

The application will be available at https://ipclawassitant.streamlit.app/ in your web browser.

### Training the Model

To train the model with your own data:

1. Prepare your data in the same JSON format as the example `ipc_data.json`
2. Run the training script:
```
python train.py --data data/ipc_data.json --epochs 20 --batch_size 32 --hidden_size 256
```

Training parameters:
- `--data`: Path to the JSON data file
- `--save_dir`: Directory to save model files (default: 'models')
- `--hidden_size`: Hidden size of LSTM (default: 256)
- `--layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.2)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 20)

## Model Architecture

The model uses a sequence-to-sequence architecture with LSTM (Long Short-Term Memory) networks:

1. **Encoder**: Processes the input text (crime description) and encodes it into a context vector
2. **Decoder**: Generates the appropriate IPC information based on the encoded context

The training process includes:
- Text preprocessing and tokenization
- Vocabulary creation
- Sequence padding and batching
- Encoder-decoder training with teacher forcing

## Sample Data Format

The application uses data in the following JSON format:

```json
[
  {
    "crime_description": "He broke into a house at night and stole valuable items.",
    "ipc_section": "457, 380",
    "ipc_title": "Lurking house-trespass or house-breaking by night + Theft",
    "ipc_description": "Section 457: House-breaking by night. Section 380: Theft in dwelling house.",
    "example_case": "State of Tamil Nadu vs A. Manickam, 2002",
    "verdict_summary": "Accused convicted for night-time burglary and theft under IPC 457 and 380."
  },
  ...
]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit, PyTorch, and NLTK
- Sample IPC data compiled from legal resources
