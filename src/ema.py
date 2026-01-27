import torch

class EMA:
    """
    Exponential Moving Average (EMA) for Teacher-Student training.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register ALL parameters of the model to track."""
        for name, param in self.model.named_parameters():
            # We track everything, even if requires_grad is False
            self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update the EMA weights using the current student model weights."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply the EMA weights to the model (turn it into the Teacher)."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (if needed)."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}