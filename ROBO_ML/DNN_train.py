import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the Dataset
class SimpleDataset(Dataset):
    def __init__(self, csv_file, input_size, output_size):
        self.data = pd.read_csv(csv_file).values
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.data[index, :self.input_size]
        targets = self.data[index, self.input_size:self.input_size + self.output_size]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Define the DNN Model
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNNModel, self).__init__()
        layers = []
        in_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())  # Activation function
            in_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Parameters
csv_file = 'D:\Research\ROBO_ML\data_nmpc.csv'  # Path to your CSV file
batch_size = 5          # Number of samples per batch
input_size = 36          # Number of input features
output_size = 53         # Number of output features
hidden_sizes = [64, 128, 64]  # Hidden layer sizes

# Dataset and DataLoader
dataset = SimpleDataset(csv_file, input_size, output_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = DNNModel(input_size, hidden_sizes, output_size)

# Loss and optimizer
criterion = nn.MSELoss()  # For regression
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)

# Training loop
num_epochs = 500

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Compute accuracy for regression
        threshold = 0.01  # Define your threshold for accuracy
        correct += (torch.abs(outputs - targets) < threshold).sum().item()
        total += targets.numel()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')




