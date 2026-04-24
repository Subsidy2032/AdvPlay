import torch
from torch.utils.data import DataLoader, TensorDataset

from advplay.ml.ops.trainers.base_trainer import BaseTrainer
from advplay.ml.models.loss_functions.registry import LOSS_FUNCTION_REGISTRY
from advplay.ml.models.optimizers.registry import OPTIMIZER_REGISTRY
from advplay.variables import available_frameworks

class PyTorchTrainer(BaseTrainer, framework=available_frameworks.PYTORCH, model=None):
    def __init__(self, X_train, y_train, config: dict = None):
        super().__init__(X_train, y_train, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None

    def preprocess_data(self):
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.long)
        return X_tensor, y_tensor

    def train(self):
        if self.model is None:
            raise NotImplementedError("Subclasses must define self.model before calling train.")

        batch_size = self.config.get("batch_size", 32)
        epochs = self.config.get("epochs", 5)
        lr = self.config.get("lr", 0.001)
        loss_name = self.config.get("loss") or "CrossEntropyLoss"
        opt_name = self.config.get("opt") or "Adam"
        weight_decay = self.config.get("weight_decay", 0.0)

        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train.squeeze(), dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if self.criterion is None:
            self.criterion = LOSS_FUNCTION_REGISTRY[available_frameworks.PYTORCH](loss_name)
        if self.optimizer is None:
            self.optimizer = OPTIMIZER_REGISTRY[available_frameworks.PYTORCH](
                opt_name, self.model, lr, weight_decay
            )

        self.model.to(self.device)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return self.model