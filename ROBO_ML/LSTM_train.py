import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Decode the output at each time step
        out = self.fc(out)
        return out
    
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length, input_size, output_size):
        self.data = pd.read_csv(csv_file).values
        self.seq_length = seq_length
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        # Number of sequences we can extract from the data
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # Get a sequence and its corresponding label
        inputs = self.data[index:index + self.seq_length, :self.input_size]
        targets = self.data[index:index + self.seq_length, self.input_size:]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)



# Parameters
csv_file = 'D:\Research\ROBO_ML\data_nmpc.csv'  # Path to your CSV file
seq_length = 10        # Sequence length for LSTM
batch_size = 128         # Number of sequences per batch
input_size = 36         # Number of input features
output_size = 53        # Number of output features


# Dataset and DataLoader
dataset = TimeSeriesDataset(csv_file, seq_length, input_size, output_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
hidden_size = 128
num_layers = 2
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()  
# criterion = torch.nn.L1Loss()
# optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-6)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, weight_decay=1e-5, momentum=0.9)


# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Compute accuracy
        if isinstance(criterion, nn.CrossEntropyLoss):  # For classification
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        else:  # For regression, compute accuracy within a threshold
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
