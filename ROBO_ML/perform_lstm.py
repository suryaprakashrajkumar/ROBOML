import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import time

# Dataset with overlapping windows
class OverlapTimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length, overlap, input_size, output_size):
        data = pd.read_csv(csv_file).values
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(data)
        self.seq_length = seq_length
        self.overlap = overlap
        self.input_size = input_size
        self.output_size = output_size
        self.scaler = scaler  # Save scaler for inverse transformation

    def __len__(self):
        return (len(self.data) - self.seq_length) // (self.seq_length - self.overlap)

    def __getitem__(self, index):
        start = index * (self.seq_length - self.overlap)
        inputs = self.data[start:start + self.seq_length, :self.input_size]
        targets = self.data[start + self.seq_length - 1, self.input_size:]
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
csv_file = 'Data\D1.csv'  # Path to your CSV file data_nmpc.csv'
seq_length =5
overlap =0
batch_size = 16
input_size = 63
output_size = 40
hidden_size = 512
num_layers = 3
num_epochs = 20000
learning_rate = 0.00001

# Dataset and DataLoader
dataset = OverlapTimeSeriesDataset(csv_file, seq_length, overlap, input_size, output_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = CombinedLoss(alpha=0.6)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)
scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast():  
            outputs = model(inputs)  # Final time step prediction
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Compute accuracy for regression (based on threshold)
        threshold = 0.01
        correct += (torch.abs(outputs - targets) < threshold).sum().item()
        total += targets.numel()

    scheduler.step()  # Step the learning rate scheduler
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")



def test_model(test_csv_file, model, batch_size=16, input_size=63, output_size=40):
    # Load test data
    test_dataset = OverlapTimeSeriesDataset(test_csv_file, seq_length, overlap, input_size, output_size)
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

            # Mixed precision for testing as well
            with torch.cuda.amp.autocast():
                outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Collect predictions and true values for later analysis
            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.cpu().numpy())

            # Compute accuracy for regression (based on threshold)
            threshold = 0.01
            correct += (torch.abs(outputs - targets) < threshold).sum().item()
            total += targets.numel()
    end_time = time.time()  # Record the end time
    time_taken = end_time - start_time  # Calculate the time taken for the prediction
    avg_loss = total_loss / len(test_dataloader)
    accuracy = 100 * correct / total

    # Flatten lists of predictions and true values
    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return predictions, true_values, time_taken

# Example usage
test_csv_file = 'Data\D2.csv'  # Path to your test data CSV file data_nmpc_test.csv
predictions, true_values, time_taken = test_model(test_csv_file, model)
print(f"Time taken for prediction: {time_taken:.6f} seconds")
# Optionally, visualize predictions vs true values
plt.figure()
plt.plot(true_values[:200,1], label='True Values')  # First 100 samples
plt.plot(predictions[:200,1], label='Predictions')  # First 100 predictions
plt.legend()
plt.show()

# Example usage
test_csv_file = 'Data\D12.csv'  # Path to your test data CSV file data_nmpc_test.csv
predictions, true_values, time_taken = test_model(test_csv_file, model)
print(f"Time taken for prediction: {time_taken:.6f} seconds")
# Optionally, visualize predictions vs true values
plt.figure()
plt.plot(true_values[:200,1], label='True Values')  # First 100 samples
plt.plot(predictions[:200,1], label='Predictions')  # First 100 predictions
plt.legend()
plt.show()

# Example usage
test_csv_file = 'Data\D1.csv'  # Path to your test data CSV file data_nmpc_test.csv
predictions, true_values, time_taken = test_model(test_csv_file, model)
print(f"Time taken for prediction: {time_taken:.6f} seconds")
# Optionally, visualize predictions vs true values
plt.figure()
plt.plot(true_values[:200,1], label='True Values')  # First 100 samples
plt.plot(predictions[:200,1], label='Predictions')  # First 100 predictions
plt.legend()
plt.show()