import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Dataset Class for TimeSeries
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_files, seq_length, input_size, output_size, scaler=None):
        self.data = []
        self.seq_length = seq_length
        self.input_size = input_size
        self.output_size = output_size
        self.scaler = scaler if scaler else MinMaxScaler()

        # Load all data for consistent scaling
        raw_data_list = [pd.read_csv(csv_file).values for csv_file in csv_files]
        all_data = np.concatenate(raw_data_list, axis=0)  # Merge datasets for uniform scaling

        if scaler is None:
            self.scaler.fit(all_data)

        for raw_data in raw_data_list:
            scaled_data = self.scaler.transform(raw_data)

            # Creating input-output pairs
            for i in range(len(scaled_data) - seq_length):  # Adjusted to consider sequence of length `seq_length`
                # Inputs: sequence of `seq_length` time steps (t0 to t4)
                inputs = scaled_data[i:i + seq_length, :input_size]
                # Targets: the output values at time step t5
                targets = scaled_data[i + seq_length - 1, input_size:]  # Predicting t4
                self.data.append((inputs, targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs, targets = self.data[index]
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


# Training loop
def train_model(csv_files, model, dataloader, criterion, optimizer, scheduler, num_epochs=20000, log_file="training_log.txt"):
    # Open log file
    with open(log_file, "w") as log:
        log.write("Epoch,Loss,Accuracy\n")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass and loss calculation
            outputs = model(inputs)  # Final time step prediction
            loss = criterion(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy for regression (based on threshold)
            threshold = 0.01
            correct += (torch.abs(outputs - targets) < threshold).sum().item()
            total += targets.numel()

        scheduler.step()  # Step the learning rate scheduler
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total

        # Logging results
        with open(log_file, "a") as log:
            log.write(f"{epoch+1},{avg_loss:.4f},{accuracy:.2f}\n")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


# Testing function
def test_model(test_csv_files, model, batch_size=16, input_size=63, output_size=40):
    test_dataset = TimeSeriesDataset(test_csv_files, seq_length, input_size, output_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    true_values = []
    start_time = time.time()  # Record the start time

    with torch.no_grad():  # Disable gradient calculation for testing
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  # Forward pass

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.cpu().numpy())

            # Compute accuracy for regression
            threshold = 0.01
            correct += (torch.abs(outputs - targets) < threshold).sum().item()
            total += targets.numel()

    end_time = time.time()  # Record the end time
    time_taken = end_time - start_time  # Calculate the time taken for prediction
    avg_loss = total_loss / len(test_dataloader)
    accuracy = 100 * correct / total

    # Flatten predictions and true values for analysis
    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return predictions, true_values, time_taken


# Set Parameters
csv_files = [f"Data/D{i}.csv" for i in range(1, 19)]  # Files from D1 to D18
test_csv_files = ["Data/D19.csv", "Data/D20.csv"]  # Validation data
seq_length = 5
input_size = 63
output_size = 40
hidden_size = 512
num_layers = 3
num_epochs = 20000
learning_rate = 0.00001
batch_size = 128

# Initialize DataLoader
dataset = TimeSeriesDataset(csv_files, seq_length, input_size, output_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = CombinedLoss(alpha=0.6)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train model
train_model(csv_files, model, dataloader, criterion, optimizer, scheduler, num_epochs)

# Save the model if validation accuracy improves
best_val_accuracy = 0
for epoch in range(num_epochs):
    # Run validation after every epoch
    predictions, true_values, _ = test_model(test_csv_files, model)
    val_accuracy = 100 * np.mean(np.abs(predictions - true_values) < 0.01)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")  # Save the model with best validation accuracy

# Final test
predictions, true_values, time_taken = test_model(test_csv_files, model)
print(f"Final Test Time: {time_taken:.6f} seconds")

# Visualization
plt.figure()
plt.plot(true_values[:200, 1], label="True Values")  # First 200 true values
plt.plot(predictions[:200, 1], label="Predictions")  # First 200 predictions
plt.legend()
plt.show()
