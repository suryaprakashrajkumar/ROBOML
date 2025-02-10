import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Dataset class with normalization
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length, input_size, output_size):
        # Load and normalize data
        data = pd.read_csv(csv_file).values
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(data)
        self.seq_length = seq_length
        self.input_size = input_size
        self.output_size = output_size
        self.scaler = scaler  # Save scaler for inverse transform if needed later

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        inputs = self.data[index:index + self.seq_length, :self.input_size]
        targets = self.data[index:index + self.seq_length, self.input_size:]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# LSTM model with Layer Normalization
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Apply layer normalization
        out = self.layer_norm(out)
        # Decode the output at each time step
        out = self.fc(out)
        return out

# Parameters
csv_file = 'D:\Research\ROBO_ML\data_nmpc.csv'  # Path to your CSV file
seq_length = 30       # Sequence length for LSTM
batch_size = 5      # Number of sequences per batch
input_size = 36       # Number of input features
output_size = 53      # Number of output features

# Dataset and DataLoader
dataset = TimeSeriesDataset(csv_file, seq_length, input_size, output_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model parameters
hidden_size = 128
num_layers = 5
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
# criterion  nn.MSELoss()

criterion = nn.HuberLoss(delta=100.0)

optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, weight_decay=1e-5, momentum=0.9)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute accuracy
        threshold = 0.01  # Adjust threshold as per normalized data range
        correct += (torch.abs(outputs - targets) < threshold).sum().item()
        total += targets.numel()

        total_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader) * 10
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
