import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the model
class REITModel(nn.Module):
    def __init__(self, input_dim):
        super(REITModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create synthetic dataset
def create_dataset():
    np.random.seed(42)
    num_samples = 1000
    X = np.random.rand(num_samples, 4)  # Features: Example data
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * np.random.rand(num_samples)  # Target
    return X, y

X, y = create_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch Dataset and DataLoader
class REITDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = REITDataset(X_train, y_train)
test_dataset = REITDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Initialize model, loss function, and optimizer
model = REITModel(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Train the model
train_model(model, train_loader, epochs=3)

# After training the model
torch.save(model.state_dict(), "reit_model.pth")
print("Model saved as 'reit_model.pth'")


# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            predictions.extend(y_pred.numpy())
            actuals.extend(y_batch.numpy())
    return predictions, actuals

predictions, actuals = evaluate_model(model, test_loader)

# Print sample predictions and actuals
print("Sample Predictions vs Actuals:")
for i in range(5):
    print(f"Prediction: {predictions[i]:.4f}, Actual: {actuals[i]:.4f}")
