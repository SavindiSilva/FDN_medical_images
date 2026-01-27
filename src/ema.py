import torch

class EMA:
    """
    Exponential Moving Average (EMA) for Teacher-Student training.
    The Teacher model weights are a moving average of the Student model weights.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register current parameters of the model to track."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update the EMA weights using the current student model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply the EMA weights to the model (turn it into the Teacher)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (if needed)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}