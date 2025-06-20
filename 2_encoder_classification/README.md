# BERT-Based Entity Matching with HuggingFace Transformers

This module demonstrates how to build, train, and use a BERT-based classification model for entity matching using HuggingFace's transformers library. It's designed as a learning resource to understand the key components of transformer-based NLP models. We use PyTorch as the backend for the model.

## Overview

Entity matching is the task of determining whether two text entities refer to the same real-world object. This implementation uses a BERT encoder with a classification head to predict whether two product descriptions match (binary classification: 0 = different products, 1 = same product). It is based upon the [Ditto](https://github.com/megagonlabs/ditto) architecture, with slight modifications to make it simpler.

## Architecture

The model follows this architecture:
```
Input: "Product A [SEP] Product B"
    ↓
BERT Encoder (contextualized embeddings)
    ↓
Pooler Output (CLS token representation)
    ↓
Dense Layer (768 → 256) + ReLU
    ↓
Output Layer (256 → 1)
    ↓
Sigmoid (probability of match)
```

## File Structure

```
├── dataset.py              # Dataset class for loading and preprocessing data
├── model.py                # BERT classification model definition
├── trainer.py              # Training orchestration and utilities
├── utils.py                # Helper utilities (early stopping)
├── run_training.py         # Main training script
```

## Core Components

### 1. Dataset (`dataset.py`)

The `MatchingDataset` class handles data loading and preprocessing:

```python
class MatchingDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        # Loads tab-separated data: entity1\tentity2\tlabel
```

**Key features:**
- **Flexible input**: Accepts file paths or list of lines
- **Robust parsing**: Handles malformed lines gracefully
- **Tokenization**: Combines entity pairs with `[SEP]` separator
- **Dynamic padding**: Efficient batching with `pad()` static method

**Data format expected:**
```
Apple iPhone 13	Apple iPhone 13 Pro	0
Samsung Galaxy S21	Samsung Galaxy S21	1
```

### 2. Model (`model.py`)

The `MatchingModel` implements a BERT-based binary classifier. We use `bert_uncased_L-4_H-256_A-4`, which is a smaller BERT variant.

```python
class MatchingModel(torch.nn.Module):
    def __init__(self, model_name, num_labels=1):
        # BERT encoder + classification head
```

**Architecture details:**
- **BERT backbone**: Uses HuggingFace's `AutoModel` for flexibility
- **Classification head**: Dense(256→256) + ReLU + Dense(256→1)
- **Pooling**: Uses BERT's `pooler_output` (CLS token)
- **Flexible initialization**: Accepts model names or pre-loaded models

### 3. Training (`trainer.py`)

The `Trainer` class orchestrates the training process with professional ML practices:

**Key features:**
- **Configurable training**: `TrainingConfig` dataclass for hyperparameters
- **Optimization**: AdamW optimizer with linear warmup scheduling
- **Regularization**: Gradient clipping and weight decay
- **Early stopping**: Prevents overfitting
- **Logging**: Comprehensive training logs and progress bars
- **Model checkpointing**: Saves best model based on validation loss

**Training configuration:**
```python
@dataclass
class TrainingConfig:
    num_epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.1
    patience: int = 2  # for early stopping
```

### 4. Utilities (`utils.py`)

Contains helper classes like `EarlyStopping` for training utilities.

## Usage

### Basic Training

```python
# 1. Setup tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
model = MatchingModel(model_name="google/bert_uncased_L-4_H-256_A-4")

# 2. Load data
train_dataset = MatchingDataset(path="train.txt", tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=MatchingDataset.pad)

# 3. Train
trainer = Trainer(model, TrainingConfig(), torch.nn.BCEWithLogitsLoss(), device)
trainer.fit(train_loader, val_loader, "model.pth")
```

### Running the Complete Pipeline

```bash
python run_training.py
```

This script:
1. Downloads training/validation data from the Ditto dataset
2. Creates datasets and data loaders
3. Initializes the model and trainer
4. Runs training with validation

## Data Processing Pipeline

### Input Processing
1. **Text concatenation**: Combine entities with separator
   ```python
   combined = left + " [SEP] " + right
   ```

2. **Tokenization**: Convert to BERT input format
   ```python
   tokens = tokenizer(combined, max_length=512, truncation=True, return_tensors="pt")
   ```

3. **Batching**: Dynamic padding for efficient training
   ```python
   # Pad sequences to max length in batch (not global max)
   max_len = max(len(s) for s in batch_sequences)
   ```

### Model Forward Pass
1. **BERT encoding**: Get contextualized representations
2. **Pooling**: Extract CLS token representation
3. **Classification**: Pass through dense layers
4. **Loss calculation**: BCEWithLogitsLoss (includes sigmoid)

## Key Learning Concepts

### 1. **Transformer Tokenization**
- Understanding special tokens (`[CLS]`, `[SEP]`, `[PAD]`)
- Handling variable-length sequences
- Attention masks for padded tokens

### 2. **Transfer Learning**
- Using pre-trained BERT weights
- Fine-tuning vs. feature extraction

### 3. **Efficient Training**
- Dynamic batching and padding
- Gradient accumulation and clipping
- Learning rate scheduling
- Early stopping for regularization

### 4. **Production Considerations**
- Model checkpointing and versioning
- Logging and monitoring
- Error handling and data validation
- Modular code organization

## Hyperparameter Guide

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `learning_rate` | 3e-5 | AdamW learning rate | Start with 2e-5 to 5e-5 for BERT |
| `batch_size` | 8 | Training batch size | Increase if GPU memory allows |
| `max_len` | 512 | Maximum sequence length | Balance between coverage and efficiency |
| `warmup_ratio` | 0.1 | Learning rate warmup proportion | 0.1 is standard for transformers |
| `weight_decay` | 1.0 | L2 regularization strength | Reduce if underfitting |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size` in `TrainingConfig`
   - Reduce `max_len` in dataset initialization
   - Use gradient accumulation

2. **Slow training**
   - Increase `batch_size` if memory allows
   - Use multiple workers: `num_workers=4` in DataLoader
   - Consider mixed precision training

3. **Poor convergence**
   - Adjust learning rate (try 2e-5 or 5e-5)
   - Increase warmup steps
   - Check data quality and class balance

### Performance Optimization

```python
# For faster training
train_loader = DataLoader(
    dataset,
    batch_size=16,  # Increase if GPU memory allows
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

## Learning Exercises

1. **Experiment with different BERT variants**:
   - Try `distilbert-base-uncased` (smaller, faster)
   - Try `bert-large-uncased` (larger, potentially better performance)

2. **Modify the architecture**:
   - Add dropout layers
   - Try different pooling strategies (mean pooling vs. CLS)
   - Experiment with different classification head sizes

3. **Advanced techniques**:
   - Implement focal loss for imbalanced datasets
   - Add class weights to the loss function
   - Try different optimization strategies (Lion, AdaFactor)

4. **Evaluation improvements**:
   - Add precision, recall, F1-score metrics
   - Implement confusion matrix visualization
   - Add model interpretation tools (attention visualization)

## Further Reading

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [BERT Paper: "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- [Entity Matching with Deep Learning](https://github.com/megagonlabs/ditto)
- [PyTorch Training Best Practices](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Ditto Paper](https://arxiv.org/pdf/2004.00584)
