import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import Union, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from time import time
from tqdm import tqdm

from utils import EarlyStopping
from model import MatchingModel


@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters.

    Defines all hyperparameters and settings needed for model training,
    including optimization, regularization, and logging parameters.

    Attributes:
        num_epochs: Maximum number of training epochs.
        batch_size: Number of samples per training batch.
        learning_rate: Initial learning rate for AdamW optimizer.
        warmup_ratio: Proportion of training steps for learning rate warmup.
        patience: Number of epochs to wait before early stopping.
        max_grad_norm: Maximum gradient norm for clipping.
        weight_decay: L2 regularization strength.
        save_best_only: Whether to save only the best model checkpoint.
        log_dir: Directory path for saving training logs.
    """

    num_epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.1
    patience: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 1.0
    save_best_only: bool = True
    log_dir: str = "logs"


class Trainer:
    """Orchestrates the model training process.

    Handles all aspects of training including optimization, validation,
    early stopping, logging, and model checkpointing. Provides a clean
    interface for training BERT-based models.

    Attributes:
        model: The MatchingModel to train.
        config: TrainingConfig with hyperparameters.
        loss_fn: Loss function for training.
        device: Device (CPU/GPU) for training.
        training_history: Dictionary storing training metrics.
        logger: Logger instance for training logs.
    """

    def __init__(
        self,
        model: MatchingModel,
        config: TrainingConfig,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: The MatchingModel instance to train.
            config: TrainingConfig object containing hyperparameters.
            loss_fn: PyTorch loss function (e.g., BCEWithLogitsLoss).
            device: Device to run training on (CPU or CUDA).

        Note:
            The model is automatically moved to the specified device.
            Logging is configured to write to both file and console.
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(self.device)
        self.training_history = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the training process.

        Sets up logging to write training progress and metrics to both
        a log file and the console. Creates the log directory if it
        doesn't exist.

        Note:
            Log files are saved as '{log_dir}/bert_training.log'.
            Log level is set to INFO to capture training progress.
        """
        Path(self.config.log_dir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"{self.config.log_dir}/bert_training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _create_optimizer_scheduler(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
        """Create optimizer and learning rate scheduler.

        Sets up AdamW optimizer with linear warmup scheduler for stable
        training of transformer models.

        Args:
            num_training_steps: Total number of training steps across all epochs.

        Returns:
            tuple: (optimizer, scheduler) pair configured for training.

        Note:
            Uses AdamW optimizer with epsilon=1e-8 and configurable weight decay.
            Linear warmup helps stabilize early training phases.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
    ) -> Tuple[float, float]:
        """Train the model for one epoch.

        Performs forward pass, loss computation, backpropagation, and parameter
        updates for all batches in the training set. Includes gradient clipping
        and progress tracking.

        Args:
            train_loader: DataLoader containing training batches.
            optimizer: PyTorch optimizer for parameter updates.
            scheduler: Learning rate scheduler.
            epoch: Current epoch number (0-indexed).

        Returns:
            tuple: (average_loss, average_accuracy) for the epoch.

        Note:
            Model is set to training mode. Gradients are clipped to prevent
            exploding gradients. Progress is displayed with tqdm.
        """
        self.model.train()
        total_loss, total_accuracy = 0.0, 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Training]"
        )
        for batch in progress_bar:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }

            labels = batch["labels"].float()
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs.squeeze(), labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            accuracy = (preds == labels).float().mean()
            total_accuracy += accuracy.item()

            progress_bar.set_postfix(
                loss=total_loss / (progress_bar.n + 1),
                acc=total_accuracy / (progress_bar.n + 1),
            )

        return total_loss / len(train_loader), total_accuracy / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model for one epoch.

        Evaluates model performance on validation set without updating parameters.
        Computes loss and accuracy metrics for monitoring training progress.

        Args:
            val_loader: DataLoader containing validation batches.

        Returns:
            tuple: (average_loss, average_accuracy) for the validation set.

        Note:
            Model is set to evaluation mode and gradients are disabled for efficiency.
            Uses the same loss function and accuracy computation as training.
        """
        self.model.eval()
        total_loss, total_accuracy = 0.0, 0.0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch in progress_bar:
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }

                labels = batch["labels"].float()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.squeeze(), labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                accuracy = (preds == labels).float().mean()
                total_accuracy += accuracy.item()

                progress_bar.set_postfix(
                    loss=total_loss / (progress_bar.n + 1),
                    acc=total_accuracy / (progress_bar.n + 1),
                )

        return total_loss / len(val_loader), total_accuracy / len(val_loader)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, model_save_path: str
    ) -> Dict[str, List[Union[int, float]]]:
        """Train the model with validation monitoring.

        Executes the complete training loop including validation, early stopping,
        and model checkpointing. Tracks training history and logs progress.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            model_save_path: Path where the best model checkpoint will be saved.

        Returns:
            dict: Training history containing metrics for each epoch:
                - epoch: List of epoch numbers
                - epoch_time: List of training times per epoch
                - train_loss: List of training losses
                - train_acc: List of training accuracies
                - val_loss: List of validation losses
                - val_acc: List of validation accuracies

        Note:
            Saves the model with the best validation loss. Training can stop
            early if validation loss doesn't improve for 'patience' epochs.
        """
        self.logger.info("Starting training process...")
        self.logger.info(f"Device: {self.device}")

        num_training_steps = len(train_loader) * self.config.num_epochs
        optimizer, scheduler = self._create_optimizer_scheduler(num_training_steps)
        early_stopping = EarlyStopping(patience=self.config.patience)

        best_val_loss = float("inf")
        history = {
            "epoch": [],
            "epoch_time": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(self.config.num_epochs):
            start_time = time()
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, scheduler, epoch
            )
            val_loss, val_acc = self._validate_epoch(val_loader)
            epoch_time = time() - start_time

            self.logger.info(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s"
            )

            history["epoch"].append(epoch + 1)
            history["epoch_time"].append(epoch_time)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.logger.info(
                    f"New best validation loss: {val_loss:.4f}. Saving model..."
                )
                torch.save(self.model.state_dict(), model_save_path)

            if early_stopping(val_loss):
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info("Training completed!")
        self.training_history = history
        return history
