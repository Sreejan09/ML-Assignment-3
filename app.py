
import streamlit as st
import torch 
import os
import random
import torch.nn as nn
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Step 1: Load the vocabulary
def load_vocab(vocab_path):
    # Load the vocabulary from the file
    vocab = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            word, idx = line.strip().split('\t')  # Assuming tab-separated values
            vocab[word] = int(idx)  # Store as integer index
    return vocab

# Load vocabulary
stoi = load_vocab('stoi.txt')  # Adjust this path as necessary

# Create idx2word mapping
itos = {idx: word for word, idx in stoi.items()}

# Calculate vocabulary size
vocab_size = len(stoi)
model_dir = "model"

# Dropdowns for user criteria
embedding_size = st.selectbox("Select embedding size:", [64, 128])
context_length = st.selectbox("Select context length:", [5, 10, 15])
activation_function = st.selectbox("Select activation function:", ["ReLU", "tanh"])
hidden_dim=1024
# Additional inputs
user_text = st.text_input("Enter text for model to process (alphanumeric):")
prediction_length = st.number_input("Enter the number of words to predict:", min_value=1, max_value=100, step=1)

# Mapping activation function to lower case to match model naming convention
activation_fn = activation_function.lower()

# Construct the model filename based on user criteria
def prediction_words(model, user_text, prediction_length, context_length):
    words = random.sample(list(stoi.keys()), prediction_length)
    for i in words:
        user_text = user_text + " " + i

    return user_text



class NextWord(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.relu(self.lin1(x))
    x = self.lin2(x)
    return x


model_filename = f"model_embedding{embedding_size}_hidden_1024_{activation_fn}_context{context_length}.pth"
model_path = os.path.join(model_dir, model_filename)


import difflib

def replace_oov_words(input_text):
    processed_words = []
    for word in input_text.split():
        if word in stoi:
            processed_words.append(word)
        else:
            # Find closest match
            similar_words = difflib.get_close_matches(word, list(stoi.keys()), n=1)
            replacement = similar_words[0] if similar_words else '<OOV>'
            processed_words.append(replacement)
    return processed_words

def predict_next_words(model, input_text, k, context_length):
    processed_words = replace_oov_words(input_text)
    input_indices = [stoi[word] for word in processed_words[-context_length:]]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_tensor)
    top_k_indices = torch.topk(predictions[0, :], k).indices
    return [itos[idx.item()] for idx in top_k_indices]


if st.button("Predict"):
    try:
        model=torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()  # Switch to evaluation mode
    except RuntimeError as e:
        st.error(f"Error loading model: {e}")

    
    prediction = prediction_words(model, user_text, prediction_length, context_length)
    st.write(prediction)