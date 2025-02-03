"""
Encoder only models, like BERT, excel at tasks like classification 
and retrieval augmented generation (RAG). Encoders are one part of a 
Transformer, with the other part being a decoder. Below are two
examples- one in which we use ModernBERT to generate embeddings
and another where we build a classification model using BERT
and TensorFlow.

"""

###############################
# Example 1: Generate embeddings
###############################

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer, BertModel

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
model = BertModel.from_pretrained("google/bert_uncased_L-4_H-256_A-4")

text_inputs = "John Doe lives in Nashville, TN."

tokenized_inputs = tokenizer(
    text=text_inputs,
    return_tensors="pt",  # Can return np, tf, or pt
    padding="max_length",
    truncation=True,
    max_length=128,
)

outputs = model(**tokenized_inputs)

"""
There are two outputs:
    * last_hidden_state: Embeddings for each token
    * pooler_output: Represents the entire input sequence
"""

outputs["last_hidden_state"]
outputs["pooler_output"]
