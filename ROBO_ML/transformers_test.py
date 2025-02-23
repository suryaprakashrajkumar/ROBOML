import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import time

# ---------------------------
# Dataset Loader
# ---------------------------
class RegressionDataset(Dataset):
    def __init__(self, file_list, input_size, output_size, scaler=None):
        self.data = []
        self.input_size = input_size
        self.output_size = output_size
        self.scaler = MinMaxScaler()
        
        for file in file_list:
            df = pd.read_csv(file).values
            self.data.append(df)

        self.data = np.concatenate(self.data, axis=0)  # Stack all data
        self.data = self.scaler.fit_transform(self.data)
        # if scaler:
        #     self.data = scaler.transform(self.data)
        # else:
        #     self.scaler = None  # No scaling used

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.data[index, :self.input_size]
        targets = self.data[index, self.input_size:self.input_size + self.output_size]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# ---------------------------
# Transformer Model
# ---------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.4):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x).squeeze(1)  # Remove sequence dimension
        return self.fc_out(x)

# ---------------------------
# Training and Validation Function
# ---------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, log_file="training_log_transformers.txt"):
    best_val_acc = 0.0
    model_save_path = "best_model.pth"

    # Open log file
    with open(log_file, "w") as log:
        log.write("Epoch, Train Loss, Train Acc, Val Loss, Val Acc\n")

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Accuracy calculation (within a threshold)
            total_loss += loss.item()
            correct += (torch.abs(outputs - targets) < 0.01).sum().item()
            total += targets.numel()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- Validation ---
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                correct += (torch.abs(outputs - targets) < 0.01).sum().item()
                total += targets.numel()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

        # --- Logging ---
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        print(log_msg)
        with open(log_file, "a") as log:
            log.write(f"{epoch}, {train_loss:.4f}, {train_acc:.2f}, {val_loss:.4f}, {val_acc:.2f}\n")

        scheduler.step()

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at {model_save_path}")

# ---------------------------
# Main Execution
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
train_files = [f"Data/D{i}.csv" for i in range(1, 19)]  # Train on D1-D18
val_files = ["Data/D19.csv", "Data/D20.csv"]  # Validate on D19-D20

# Hyperparameters
input_size = 63
output_size = 40
batch_size = 64
num_epochs = 50
learning_rate = 0.000001

# Load data
train_dataset = RegressionDataset(train_files, input_size, output_size)
val_dataset = RegressionDataset(val_files, input_size, output_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = TransformerModel(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
