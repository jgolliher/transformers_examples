class EarlyStopping:
    """Early stopping utility to prevent overfitting.

    Monitors validation loss and stops training when improvement stagnates.
    Helps prevent overfitting and reduces unnecessary computation.

    Attributes:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change in loss to qualify as improvement.
        counter: Current count of epochs without improvement.
        best_loss: Best validation loss seen so far.
    """

    def __init__(self, patience: int = 2, min_delta: float = 0.0) -> None:
        """Initialize the EarlyStopping monitor.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
                Higher values allow more time for recovery from local minima.
            min_delta: Minimum change in monitored metric to qualify as improvement.
                Helps ignore trivial fluctuations in validation loss.

        Note:
            Counter starts at 0 and best_loss starts at infinity.
            Training continues as long as counter < patience.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss.

        Compares current validation loss to the best seen so far.
        Updates internal state and returns whether to stop training.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            bool: True if training should stop (early stopping triggered),
                False if training should continue.

        Note:
            Improvement is defined as: val_loss < (best_loss - min_delta).
            Counter resets to 0 when improvement is detected.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
