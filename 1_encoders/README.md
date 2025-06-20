# Encoder-Only Models: Generating Embeddings with BERT

## Overview

Encoder-only models, such as BERT, are powerful tools for tasks like text classification and retrieval-augmented generation (RAG). These models focus on understanding input text by generating embeddings that capture semantic meaning. We will explore how to use BERT to generate embeddings for text inputs.


---

## Key Concepts

### Tokenization
Before feeding text into a BERT model, it must be tokenized. Tokenization involves:
- Splitting the input text into smaller units (tokens).
- Mapping tokens to unique IDs using a vocabulary.
- Padding and truncating sequences to ensure uniform length.

### Outputs from BERT
BERT provides two primary outputs:
1. **Last Hidden State**: A tensor containing embeddings for each token in the input sequence. Shape: `(batch_size, sequence_length, hidden_size)`
2. **Pooler Output**: A tensor representing the embedding for the entire input sequence. Shape: `(batch_size, hidden_size)`

---

## Code Walkthrough

### Example 1: Single Input Embedding

#### Loading the Model and Tokenizer
We use the `AutoTokenizer` and `BertModel` classes from HuggingFace's `transformers` library to load a pre-trained BERT model and its tokenizer.

```python
from transformers import AutoTokenizer, BertModel

MODEL_PATH = "path_to_your_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
```

#### Tokenzing Input Text
The input text is tokenized using the `tokenizer` object. We specify parameters like `padding`, `truncation`, and `max_length` to ensure compatibility with the model.

```python
text_inputs = "John Doe lives in Nashville, TN."
tokenized_inputs = tokenizer(
    text=text_inputs,
    return_tensors="pt",  # Return PyTorch tensors
    padding="max_length",  # Pad to max length
    truncation=True,  # Truncate if input exceeds max length
    max_length=512,  # Typical max length for BERT models
)
```

#### Generate Embeddings
The tokenized inputs are passed through the model to generate embeddings. The model returns two outputs:

* `last_hidden_state`: Embeddings for each token.
* `pooler_output`: Embedding for the entire input sequence.

```python
outputs = model(**tokenized_inputs)
last_hidden_state = outputs["last_hidden_state"]
pooler_output = outputs["pooler_output"]
```

#### Output Shapes
The shapes of the outputs depend on the model configuration:

* `last_hidden_state`: (batch_size, sequence_length, hidden_size)
* `pooler_output`: (batch_size, hidden_size)

#### Example Output
For the input text "John Doe lives in Nashville, TN.", the shapes of the outputs might look like:

* las`t_hidden_state: (1, 512, 256) (assuming hidden_size=256)
* `pooler_output`: (1, 256)

### Example 2: Batch Processing for Efficiency
When working with a large list of text inputs, batch processing is more efficient. The `transformers` library supports batch encoding, which tokenizes multiple inputs simulataneously.

```python
texts = [
    "John Doe lives in Nashville, TN.",
    "Google invented the transformer architecture.",
    "Transformers are revolutionizing natural language processing.",
    "HuggingFace provides state-of-the-art NLP models.",
]
batch_tokenized_inputs = tokenizer(
    text=texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)
batch_outputs = model(**batch_tokenized_inputs)
batch_last_hidden_state = batch_outputs["last_hidden_state"]
batch_pooler_output = batch_outputs["pooler_output"]
```

#### Output Shapes
For batch processing, the shapes of the outputs depend on the number of inputs:

* `batch_last_hidden_state`: (batch_size, sequence_length, hidden_size)
* `batch_pooler_output`: (batch_size, hidden_size)

#### Example Output
For a batch of 4 text inputs, the shapes of the outputs might look like:

* `batch_last_hidden_state`: (4, 512, 256) (assuming hidden_size=256)
* `batch_pooler_output`: (4, 256)


## Applications

* **Text Classification**: Use `pooler_output` as input to a classifier.
* **Semantic Search**: Use `last_hidden_state` to compare token-level embeddings
* **Retrieval-Augmented Generation (RAG)**: Combine embeddings with external knowledge bases for LLMs