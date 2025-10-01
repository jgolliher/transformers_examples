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

# Version-compatible imports
try:
    # Newer PyTorch versions
    from torch.amp import autocast, GradScaler

    TORCH_AMP_AVAILABLE = True
except ImportError:
    # Older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler

    TORCH_AMP_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters.

    Defines all hyperparameters and settings needed for model training,
    including optimization, regularization, and logging parameters.

    Attributes:
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        learning_rate (float): Initial learning rate for AdamW optimizer.
        warmup_ratio (float): Proportion of training steps for learning rate warmup.
        patience (int): Number of epochs to wait before early stopping.
        max_grad_norm: Maximum gradient norm for clipping.
        weight_decay (float): L2 regularization strength.
        save_best_only (bool): Whether to save only the best model checkpoint.
        log_dir (str): Directory path for saving training logs.
        use_mixed_precision (bool): Whether to use automatic mixed precision
            training. Defaults to True.
        mixed_precision_dtype (str):  The desired dtype for mixed precision.
            ('fp16' or 'bf16'). Defaults to 'fp16'.
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
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "fp16"


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

        # --- Device Error Handling ---
        # If device is MPS and PyTorch version doesn't support it, raise error
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                f"Device is set to 'mps' but the MPS backend is not available."
            )

        # --- Setup mixed precision ---
        self.dtype = None
        self.scaler = None
        self.use_mixed_precision = self.config.use_mixed_precision and device.type in [
            "cuda",
            "mps",
        ]
        if self.use_mixed_precision:
            # Set the desired precision
            self.dtype = (
                torch.bfloat16
                if self.config.mixed_precision_dtype == "bf16"
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            print(
                f"Training on {self.device} with mixed precision ({self.config.mixed_precision_dtype}) enabled."
            )

            # GradScaler is *ONLY* for CUDA
            if self.device.type == "cuda":
                if TORCH_AMP_AVAILABLE:
                    # New API: specify device type
                    self.scaler = GradScaler(device.type)
                else:
                    # Old API: no device parameter (CUDA only)
                    self.scaler = GradScaler()
            else:
                print(f"Training on {self.device} with mixed-precision disabled.")

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
        """Execute one training epoch with mixed precision and progress tracking.

        Processes all batches in the training data loader, applying forward pass,
        loss calculation, backpropagation, and optimizer updates. Automatically
        handles mixed precision training on CUDA devices and provides real-time
        progress monitoring.


        Args:
            train_loader (DataLoader): Training data loader with labeled batches.
            optimizer (torch.optim.Optimizer): Configured AdamW optimizer.
            scheduler (Any): Learning rate scheduler for step-wise updates.
            epoch (int): Current epoch number for progress display.

        Returns:
            tuple: (average_loss, average_accuracy) for the epoch.

        Note:
            Training Features:
            - Automatic mixed precision on CUDA for memory and speed benefits
            - Gradient clipping to prevent exploding gradients
            - Real-time progress bar with loss and accuracy updates
            - Per-step learning rate scheduling
            - Proper model mode management (train mode)
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

            optimizer.zero_grad()

            # Use mixed-precision on CUDA or MPS
            if self.use_mixed_precision:
                # Version-compatible augocast usage
                if TORCH_AMP_AVAILABLE:
                    # New API: supports device_type and dtype parameters
                    with autocast(device_type=self.device.type, dtype=self.dtype):
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs.squeeze(-1), labels)
                else:
                    # Old API: Only supports enabled parameters, CUDA only
                    with autocast(enabled=True):
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs.squeeze(-1), labels)

                # GradScaler is only for CUDA
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:  # For MPS or if scaler is not used
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    optimizer.step()
            else:  # Standard training on CPU
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.squeeze(-1), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            # Check accuracy calculation
            preds = (outputs > 0).float()
            preds = (preds.squeeze() == labels).float().mean().item()
            total_accuracy += preds

            progress_bar.set_postfix(
                loss=total_loss / (progress_bar.n + 1),
                acc=total_accuracy / (progress_bar.n + 1),
            )

        return total_loss / len(train_loader), total_accuracy / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Executes one validation epoch with comprehensive metric tracking.

        Evaluates the model on the validation dataset without gradient computation,
        providing unbiased estimates of model performance. Used for early stopping
        decisions and best model selection during training.

        Args:
            val_loader: DataLoader containing validation batches.

        Returns:
            tuple: (average_loss, average_accuracy) for the validation set.

        Note:
            Validation Features:
            - Gradient computation disabled for efficiency
            - Model set to evaluation mode
            - Progress bar with real-time metrics
            - No optimizer updates or learning rate changes
            - Used for early stopping and model checkpointing decisions
        """
        self.model.eval()
        total_loss, total_accuracy = 0.0, 0.0

        with torch.inference_mode():
            progress_bar = tqdm(
                val_loader,
                desc="Validation",
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
                loss = self.loss_fn(outputs.squeeze(-1), labels)

                total_loss += loss.item()

                # Check accuracy calculation
                preds = (outputs > 0).float()
                preds = (preds.squeeze() == labels).float().mean().item()
                total_accuracy += preds

                progress_bar.set_postfix(
                    loss=total_loss / (progress_bar.n + 1),
                    acc=total_accuracy / (progress_bar.n + 1),
                )
        return total_loss / len(val_loader), total_accuracy / len(val_loader)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, model_save_path: str
    ) -> Dict[str, List[Union[int, float]]]:
        """Train the BERT model with comprehensive monitoring and optimization.

        Executes the complete training loop including optimizer setup, learning rate
        scheduling, training/validation cycles, progress tracking, model checkpointing,
        and early stopping. Provides detailed loging and retrusn comprehensive training
        history for analysis.

        Train the model with validation monitoring.

        Executes the complete training loop including validation, early stopping,
        and model checkpointing. Tracks training history and logs progress.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training data. Should use
                BERTMatchingDataset with appropriate collate function and contain labeled
                person pairs.
            val_loader (DataLoader): PyTorch DataLoader for validation data. Used for
                model evaluation, early stopping decisions, and best model selection.
            model_save_path (str): File path where the best model (lowest validation loss)
                will be saved. Should end with .pt, .pth, or .bin extension.

        Returns:
            Dict[str, List]: Training history dictionary containing lists of metrics:
                - 'epoch': Epoch numbers
                - 'epoch_time': Time taken per epoch (seconds)
                - 'train_loss': Training loss per epoch
                - 'train_acc': Training accuracy per epoch
                - 'val_loss': Validation loss per epoch
                - 'val_acc': Validation accuracy per epoch

        Note:
            Saves the model with the best validation loss. Training can stop
            early if validation loss doesn't improve for 'patience' epochs.
        """
        # Enable anamoly detection to get a stack trace on the exact operation
        # if training producws NaN gradients
        torch.autograd.set_detect_anomaly(True)

        self.logger.info("Starting training process...")
        self.logger.info(f"Device: {self.device}")
        if not self.use_mixed_precision:
            self.logger.info(f"Mixed precision disabled (CPU training)")
        else:
            self.logger.info(f"Mixed precision enabled (CUDA training)")

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
