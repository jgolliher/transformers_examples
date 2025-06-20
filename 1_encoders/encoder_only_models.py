"""
Encoder-only models, like BERT, excel at tasks such as classification
and retrieval-augmented generation (RAG). Encoders are one part of a
Transformer, with the other part being a decoder. Below is an example
of using ModernBERT to generate embeddings.

"""

import torch
from transformers import AutoTokenizer, BertModel

# Path to the pre-trained BERT model OR the name of the HuggingFace repo
MODEL_PATH = "C:/path/to/bert_uncased_L-4_H-256_A-4"
MODEL_PATH = "google/bert_uncased_L-4_H-256_A-4"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)

###############################
# Example 1: Generate embeddings for a single input
###############################

# Input text
text_inputs = "John Doe lives in Nashville, TN."

# Tokenize the input text
tokenized_inputs = tokenizer(
    text=text_inputs,
    return_tensors="pt",  # Return PyTorch tensors
    padding="max_length",  # Pad to max length
    truncation=True,  # Truncate if input exceeds max length
    max_length=512,  # Typical max length for BERT models
)

# Pass tokenized inputs through the model
outputs = model(**tokenized_inputs)

# Extract outputs
last_hidden_state = outputs["last_hidden_state"]  # Embeddings for each token
pooler_output = outputs["pooler_output"]  # Embedding for the entire sequence

# Print shapes of the outputs
print(f"Shape of last_hidden_state: {last_hidden_state.shape}")
print(f"Shape of pooler_output: {pooler_output.shape}")

###############################
# Example 2: Batch processing for large lists of text
###############################

# List of input texts
texts = [
    "John Doe lives in Nashville, TN.",
    "Google invented the transformer architecture.",
    "Transformers are revolutionizing natural language processing.",
    "HuggingFace provides state-of-the-art NLP models.",
]

# Tokenize the list of texts using batch encoding
batch_tokenized_inputs = tokenizer(
    text=texts,
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,  # Pad sequences to the longest in the batch
    truncation=True,  # Truncate sequences that exceed max length
    max_length=512,  # Typical max length for BERT models
)

# Pass batch tokenized inputs through the model
batch_outputs = model(**batch_tokenized_inputs)

# Extract outputs for the batch
batch_last_hidden_state = batch_outputs[
    "last_hidden_state"
]  # Embeddings for each token
batch_pooler_output = batch_outputs["pooler_output"]  # Embedding for each sequence

# Print shapes of the batch outputs
print(f"Shape of batch_last_hidden_state: {batch_last_hidden_state.shape}")
print(f"Shape of batch_pooler_output: {batch_pooler_output.shape}")

"""
Key Outputs:
    * last_hidden_state: A tensor containing embeddings for each token in the input.
    * pooler_output: A tensor representing the embedding for the entire input sequence.
"""
