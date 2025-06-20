import torch
from torch.utils.data import Dataset
from typing import Union, List, Dict, Any, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer


class MatchingDataset(Dataset):
    """Dataset for entity matching tasks.

    Loads and preprocesses pairs of entities with binary labels indicating whether
    they represent the same real-world object. Handles tokenization and formatting
    for BERT-based models.

    Attributes:
        tokenizer: HuggingFace tokenizer for text preprocessing.
        max_len: Maximum sequence length for tokenization.
        pairs: List of tuples containing entity pairs.
        labels: List of binary labels (0=different, 1=same).
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        max_len: int = 512,
    ) -> None:
        """Initialize the MatchingDataset.

        Args:
            path: Either a file path string containing tab-separated data or a list
                of data lines to process directly.
            tokenizer: Either a string (model name/path) to load a tokenizer or an
                already initialized tokenizer object with encode/decode methods.
            max_len: Maximum sequence length for tokenization. Sequences longer
                than this will be truncated.

        Raises:
            ValueError: If tokenizer is None.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
            self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []
        self.labels = []
        self._load_data(path)

    def _load_data(self, path: Union[str, List[str]]) -> None:
        """Load and parse dataset from file path or list of lines.

        Processes tab-separated data with format: entity1\tentity2\tlabel.
        Skips empty lines and logs warnings for malformed lines.

        Args:
            path: File path string or list of data lines to process.

        Note:
            Expected format is three tab-separated columns: left_entity, right_entity, label.
            Labels should be integers (0 or 1). Malformed lines are skipped with warnings.
        """
        if isinstance(path, list):
            lines = path
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip line, will fail silently
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            s1, s2, label = parts
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

    def __len__(self) -> int:
        """Return total number of entity pairs in dataset.

        Returns:
            int: Number of entity pairs available for training/evaluation.
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve and preprocess a single dataset item by index.

        Combines the entity pair with a separator token, tokenizes the text,
        and returns formatted tensors ready for model input.

        Args:
            idx: Index of dataset item to retrieve.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - input_ids: Tokenized input IDs tensor of shape (seq_len,).
                - attention_mask: Attention mask tensor of shape (seq_len,).
                - label: Binary matching label (0 or 1).

        Note:
            The two entities are combined as "entity1 [SEP] entity2" before tokenization.
        """
        left, right = self.pairs[idx]
        label = self.labels[idx]

        combined = left + " [SEP] " + right

        x = self.tokenizer(
            text=combined,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Output dictionary
        output = {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "label": label,
        }

        return output

    @staticmethod
    def pad(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader that pads sequences to uniform length.

        Pads all sequences in a batch to the maximum length present in the batch.
        This makes training more efficient as we do not have to pad to the max
        length of the entire dataset.

        Args:
            batch: List of dictionaries from __getitem__, each containing tokenized
                sequences and labels.

        Returns:
            Dict[str, torch.Tensor]: Padded batch dictionary containing:
                - input_ids: Padded input IDs tensor of shape (batch_size, max_len).
                - attention_mask: Padded attention mask tensor of shape (batch_size, max_len).
                - labels: Labels tensor of shape (batch_size,).

        Note:
            Padding is done with zeros. The maximum length is determined dynamically
            from the current batch, not the global dataset maximum.
        """
        # Extract data from batch
        x1_ids = [d["input_ids"] for d in batch]
        x1_masks = [d["attention_mask"] for d in batch]
        y = [d["label"] for d in batch]

        max_len = max(len(s) for s in x1_ids)

        # Pad
        x1_ids_padded = [torch.cat([s, s.new_zeros(max_len - len(s))]) for s in x1_ids]
        x1_masks_padded = [
            torch.cat([s, s.new_zeros(max_len - len(s))]) for s in x1_masks
        ]

        # Construct output dict
        padded_batch = {
            "input_ids": torch.stack(x1_ids_padded),
            "attention_mask": torch.stack(x1_masks_padded),
            "labels": torch.LongTensor(y),
        }

        return padded_batch
