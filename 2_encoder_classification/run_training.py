# Tokenizer Setup
"""Create tokenizer from pre-trained BERT model.

Uses AutoTokenizer to automatically load the appropriate tokenizer
for the specified BERT variant. Handles special tokens and vocabulary.
"""

# Training Data Loading
"""Download and prepare training dataset.

Fetches training data from the Ditto repository containing entity pairs
for computer product matching. Processes raw text into MatchingDataset.
"""

# Validation Data Loading
"""Download and prepare validation dataset.

Fetches validation data from the same source as training data.
Used for monitoring training progress and early stopping.
"""

# Model Initialization
"""Create and configure the MatchingModel.

Loads pre-trained BERT weights and adds classification head.
Configures trainer with loss function and training parameters.
"""

# Training Execution
"""Execute the complete training pipeline.

Runs training loop with validation monitoring, early stopping,
and model checkpointing. Saves the best model to disk.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import requests
from transformers import AutoTokenizer, AutoModel

from model import MatchingModel
from dataset import MatchingDataset
from trainer import Trainer, TrainingConfig

# --- 1. Define tokenizer ---
BERT_MODEL = "google/bert_uncased_L-4_H-256_A-4"

BERT_MODEL = "/Users/jgolliher/Files/BERT/bert_uncased_L-4_H-256_A-4"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

# ---  2. Load Training Data----

# Sourced from Ditto repo
url = "https://raw.githubusercontent.com/megagonlabs/ditto/refs/heads/master/data/wdc/computers/train.txt.medium"
response = requests.get(url)
if response.status_code == 200:
    # Split the content into a list of lines
    train_lines = response.text.splitlines()
    print(f"Successfully read {len(train_lines)} lines from the file.")
else:
    print(f"Failed to fetch the file. HTTP Status Code: {response.status_code}")

# Load pairs into a PyTorch dataset
train_dataset = MatchingDataset(path=train_lines, tokenizer=tokenizer, max_len=512)

# Build dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    collate_fn=MatchingDataset.pad,
)

# ---  3. Load Validation Data----
url = "https://raw.githubusercontent.com/megagonlabs/ditto/refs/heads/master/data/wdc/computers/valid.txt.medium"
response = requests.get(url)
if response.status_code == 200:
    # Split the content into a list of lines
    val_lines = response.text.splitlines()
    print(f"Successfully read {len(val_lines)} lines from the file.")
else:
    print(f"Failed to fetch the file. HTTP Status Code: {response.status_code}")

# Load pairs into PyTorch dataset
val_dataset = MatchingDataset(path=val_lines, tokenizer=tokenizer, max_len=512)

# Build dataloader
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    collate_fn=MatchingDataset.pad,
)

# --- 4. Build Model ---
bert_model = AutoModel.from_pretrained(BERT_MODEL)
model = MatchingModel(model_name=bert_model, num_labels=1)
trainer = Trainer(
    model=model,
    config=TrainingConfig(),
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    device=torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    ),
)

# --- 5. Run Training ---
trainer.fit(
    train_loader=train_loader, val_loader=val_loader, model_save_path="test.pth"
)
