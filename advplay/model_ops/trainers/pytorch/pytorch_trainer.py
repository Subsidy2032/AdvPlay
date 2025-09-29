import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from advplay.model_ops.trainers.base_trainer import BaseTrainer
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

        X_tensor = self.model.preprocess_data(self.X_train, channels_first=self.channels_first)
        y_tensor = torch.tensor(self.y_train.squeeze(), dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

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