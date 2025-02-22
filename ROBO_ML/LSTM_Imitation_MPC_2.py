import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
import os

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset class to load time-series data from multiple CSV files
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_files, seq_length, input_size, output_size):
        data_list = []
        for file in csv_files:
            data = pd.read_csv(file).values
            data_list.append(data)
        self.data = np.concatenate(data_list, axis=0)  # Concatenate all data
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)  # Normalize the entire dataset
        self.seq_length = seq_length
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # Inputs are the sequence of time steps
        inputs = self.data[index:index + self.seq_length, :self.input_size]
        # Targets are the values at t4 (last time step of the sequence)
        targets = self.data[index + self.seq_length - 1, self.input_size:]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Attention LSTM Model
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=0.3, batch_first=True)
        self.attention = nn.Linear(2 * hidden_size, 1)  # Bidirectional
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)  # Extra FC layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        x = torch.relu(self.fc1(context))
        out = self.fc2(x)
        return out

# Combined Loss Function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)

    def forward(self, y_pred, y_true):
        return self.alpha * self.mse(y_pred, y_true) + (1 - self.alpha) * self.huber(y_pred, y_true) * 100

# Parameters
seq_length = 5
input_size = 63
output_size = 40
hidden_size = 512
num_layers = 3
num_epochs = 200
learning_rate = 0.0001
batch_size = 128

# Initialize dataset and dataloaders for training and validation
train_csv_files = [f'Data/D{i}.csv' for i in range(1, 19)]  # D1 to D18 for training
val_csv_files = ['Data/D19.csv', 'Data/D20.csv']  # D19 and D20 for validation

# Initialize datasets
train_dataset = TimeSeriesDataset(train_csv_files, seq_length, input_size, output_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TimeSeriesDataset(val_csv_files, seq_length, input_size, output_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = CombinedLoss(alpha=0.6)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Set up logging
logging.info(f"Training started with {num_epochs} epochs, learning rate {learning_rate}, batch size {batch_size}")

# Training loop
best_val_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass and loss computation
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy based on t4 (compare predicted vs actual t4)
        threshold = 0.01
        correct += (torch.abs(outputs - targets) < threshold).sum().item()
        total += targets.numel()

    avg_loss = total_loss / len(train_dataloader)
    accuracy = 100 * correct / total

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Compute validation accuracy based on t4
            correct_val += (torch.abs(outputs - targets) < threshold).sum().item()
            total_val += targets.numel()

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = 100 * correct_val / total_val

    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        logging.info(f"Model saved with Validation Accuracy: {val_accuracy:.2f}%")

    scheduler.step()  # Step the learning rate scheduler

# Load the best model after training
model.load_state_dict(torch.load("best_model.pth"))
logging.info("Best model loaded for testing.")

# Testing function
def test_model(test_csv_files, model, batch_size=16, input_size=63, output_size=40):
    test_dataset = TimeSeriesDataset(test_csv_files, seq_length, input_size, output_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    true_values = []
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.cpu().numpy())

            correct += (torch.abs(outputs - targets) < 0.01).sum().item()
            total += targets.numel()

    end_time = time.time()
    time_taken = end_time - start_time
    avg_loss = total_loss / len(test_dataloader)
    accuracy = 100 * correct / total

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    logging.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    logging.info(f"Time taken for testing: {time_taken:.6f} seconds")

    return predictions, true_values, time_taken

# Test the model
test_csv_files = ['Data/D19.csv', 'Data/D20.csv']
predictions, true_values, time_taken = test_model(test_csv_files, model)

# Plotting the results
plt.figure()
plt.plot(true_values[:200, 1], label='True Values')
plt.plot(predictions[:200, 1], label='Predictions')
plt.legend()
plt.show()
