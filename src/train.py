import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train(self, dataloader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for noisy, clean in dataloader:
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")
