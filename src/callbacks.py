from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    """
    Simple early stopping on eval_loss.
    """

    def __init__(self, patience: int = 2, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_loss = logs.get("eval_loss", None)
        if current_loss is None:
            return

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            print(f"✅ Validation loss improved to {current_loss:.4f}")
        else:
            self.patience_counter += 1
            print(
                f"⏳ No improvement ({self.patience_counter}/{self.patience}) "
                f"- current: {current_loss:.4f}, best: {self.best_loss:.4f}"
            )
            if self.patience_counter >= self.patience:
                print("🛑 Early stopping triggered.")
                control.should_training_stop = True

        return control



