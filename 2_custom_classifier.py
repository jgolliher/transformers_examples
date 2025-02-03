"""This code demonstrates how to use BERT for a custom classification model in PyTorch.

It's possible to adapt this code to TensorFlow and if there is interest, I can put
that together as well. Generally speaking, Transformers (the package) is better
supported with PyTorch.

It's possible to do most of this *within* Transformers only using the
`BertForSequenceClassification`, but I think this demonstrates how 
BERT works better. 
"""

#################################
# Example 2: Classification Model
#################################

# Stacking BERT on top of your own model
import os
import torch
from torch.utils.data import Dataset, DataLoader
from time import time
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import numpy as np

# USER SPECIFICATIONS
BERT_MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"
BATCH_SIZE = 32
NUM_EPOCHS = 5
PATIENCE = 2

############### AUTOGENERATED
print(f"BERT MODEL: {BERT_MODEL_NAME}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"PATIENCE: {PATIENCE}")


# Load tokenizer and BERT model
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
bert_model.trainable = False  # Set to True to fine-tune BERT's embeddings

# Load IMDB dataset
imdb = load_dataset("imdb")


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True,
        return_token_type_ids=False,
    )


tokenized_imdb = imdb.map(preprocess_function, batched=True)

# Pull out input_ids, attention_masks, and labels
x_train_input_ids = np.array(tokenized_imdb["train"]["input_ids"])
x_train_attention_mask = np.array(tokenized_imdb["train"]["attention_mask"])
y_train = np.array(tokenized_imdb["train"]["label"])

x_test_input_ids = np.array(tokenized_imdb["test"]["input_ids"])
x_test_attention_mask = np.array(tokenized_imdb["test"]["attention_mask"])
y_test = np.array(tokenized_imdb["test"]["label"])

x_val_input_ids = np.array(tokenized_imdb["unsupervised"]["input_ids"])
x_val_attention_mask = np.array(tokenized_imdb["unsupervised"]["attention_mask"])
y_val = np.array(tokenized_imdb["unsupervised"]["label"])


# PyTorch datasets and model
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


train_dataset = CustomDataset(x_train_input_ids, x_train_attention_mask, y_train)
test_dataset = CustomDataset(x_test_input_ids, x_test_attention_mask, y_test)
val_dataset = CustomDataset(x_val_input_ids, x_val_attention_mask, y_val)

# PyTorch DataLoader
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


class ClassifierModel(torch.nn.Module):
    def __init__(self, BERT_MODEL_NAME: str = "google/bert_uncased_L-4_H-256_A-4"):
        super().__init__()
        self.bert = bert_model
        self.dense = torch.nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(256, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = bert_output.pooler_output
        dense_output = self.relu(self.dense(pooler_output))
        output = self.sigmoid(self.output(dense_output))
        return output


model = ClassifierModel(BERT_MODEL_NAME="google/bert_uncased_L-4_H-256_A-4")


# Learning Rate and Optimizer
steps_per_epoch = len(y_train) // BATCH_SIZE

num_training_steps = steps_per_epoch * NUM_EPOCHS
num_warmump_steps = int(0.1 * num_training_steps)

optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmump_steps,
    num_training_steps=num_training_steps,
)


# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
model.to(device)

best_val_loss = float("inf")
epochs_without_improvement = 0
num_batches = (len(y_train) + BATCH_SIZE - 1) // BATCH_SIZE
model_times = []

model_history = {
    "epoch": [],
    "epoch_time": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time()
    model.train()
    total_loss = 0
    total_train_acc = 0
    batch_idx = 0

    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as t:
        for batch in t:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = torch.nn.functional.binary_cross_entropy(
                outputs, labels.unsqueeze(1)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            predictions = (outputs > 0).float()
            accuracy = (predictions == labels.unsqueeze(1)).float().mean()
            total_train_acc += accuracy.item()

            # Update tqdm's postfix to show loss and accuracy
            t.set_postfix(
                {
                    "loss": total_loss / (batch_idx + 1),
                    "acc": total_train_acc / (batch_idx + 1),
                }
            )
            batch_idx += 1
    avg_train_loss = total_loss / len(train_dataloader)
    avg_train_acc = total_train_acc / len(train_dataloader)

    # Validation
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0

    with torch.no_grad():
        for batch in tqdm(test_dataset, desc=f"Validation"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.binary_cross_entropy(
                outputs, labels.unsqueeze(1)
            )
            total_val_loss += loss.item()

            predictions = (outputs > 0).float()
            accuracy = (predictions == labels.unsqueeze(1)).float().mean()
            total_val_accuracy += accuracy.item()

    avg_val_loss = total_val_loss / len(test_dataloader)
    avg_val_acc = total_val_accuracy / len(test_dataloader)

    epoch_end_time = time()
    epoch_time = epoch_end_time - epoch_start_time
    model_times.append(epoch_time)

    model_history["epoch"].append(epoch + 1)
    model_history["epoch_time"].append(epoch_time)
    model_history["train_loss"].append(avg_train_loss)
    model_history["train_acc"].append(avg_train_acc)
    model_history["val_loss"].append(avg_val_loss)
    model_history["val_acc"].append(avg_val_acc)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - train_loss: {avg_train_loss:.4f} - train_acc: {avg_train_acc:.4f} - val_loss: {avg_val_loss:.4f} - val_acc: {avg_val_acc:.4f}"
    )

    # Append information to `model_history` dict
    model_history["epoch"].append(epoch + 1)
    model_history["epoch_time"].append(epoch_end_time - epoch_start_time)
    model_history["train_loss"].append(avg_train_loss)
    model_history["train_acc"].append(avg_train_acc)
    model_history["val_loss"].append(avg_val_loss)
    model_history["val_acc"].append(avg_val_acc)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping after {epoch+1} epochs")
        break
