import torch
from transformers import AutoModel
from typing import Union, Dict


class MatchingModel(torch.nn.Module):
    """A BERT-based model for entity matching classification.

    Combines a pre-trained BERT encoder with a classification head to predict
    whether two entities refer to the same real-world object. Uses the CLS token
    representation for classification.

    Attributes:
        model_name: Name or path of the pre-trained model.
        bert: The BERT encoder model.
        dense: Hidden dense layer (768 -> 256).
        relu: ReLU activation function.
        output_layer: Final classification layer (256 -> num_labels).
        device: Current device (CPU/GPU) for the model.
        training_history: Dictionary to store training metrics.
    """

    def __init__(self, model_name: Union[str, AutoModel], num_labels: int = 1) -> None:
        """Initialize a MatchingModel.

        Args:
            model_name: The name/path of the pre-trained HuggingFace model (str),
                or an already initialized AutoModel instance.
            num_labels: The number of output labels for classification. Defaults to 1
                for binary classification.

        Raises:
            ValueError: If model_name is None.
            TypeError: If model_name is neither string nor AutoModel instance.

        Note:
            For binary classification, use num_labels=1 with BCEWithLogitsLoss.
            For multi-class classification, use num_labels=num_classes with CrossEntropyLoss.
        """
        super().__init__()
        self.model_name = model_name

        # Initialize BERT model - handle both string and instance
        if model_name is None:
            raise ValueError("model_name must be provided")
        elif isinstance(model_name, str):
            self.bert = AutoModel.from_pretrained(model_name)
        elif hasattr(model_name, "config") and hasattr(model_name, "forward"):
            self.bert = model_name
        else:
            raise TypeError(
                "model_name must be either a string (model name/path) or an initialized AutoModel."
                f"Got {type(model_name)}"
            )

        # Get the hidden size from the model's config for proper linear layer sizing
        hidden_size = self.bert.config.hidden_size

        self.dense = torch.nn.Linear(hidden_size, 256)
        self.relu = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(256, num_labels)

        # Training state
        self.device = None
        self.training_history = {}

    def forward(self, x1: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform the forward pass through the model.

        Args:
            x1: Dictionary containing tokenized input with keys:
                - 'input_ids': Token IDs tensor of shape (batch_size, seq_len).
                - 'attention_mask': Attention mask tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_labels).
                For binary classification, shape is (batch_size, 1).

        Note:
            The output logits are not passed through sigmoid/softmax. Apply
            appropriate activation in loss function or during inference.
        """
        outputs = self.bert(
            input_ids=x1["input_ids"], attention_mask=x1["attention_mask"]
        )
        pooler_output = outputs.pooler_output
        dense_output = self.relu(self.dense(pooler_output))
        output = self.output_layer(dense_output)

        return output
