import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.utils import create_clusters, human_only
from utils.data_processing import list_data
from sklearn.model_selection import  KFold
import itertools
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def predict(new_data, model, scaler):
    # Preprocess new data
    new_data_scaled = scaler.transform(new_data[['Points_X', 'Points_Y', 'Points_Z', 'Intensity']])

    # Convert to tensor
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(new_data_tensor).squeeze().numpy()

    predicted_labels = ['Human' if pred >= 0.5 else 'No Human' for pred in predictions]

    # Add predicted labels to new_data
    new_data['Predicted_Label'] = predicted_labels
    return new_data

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create Label Dataset
clusters1, points = create_clusters(list_data[0], eps=0.1)
data1 = human_only(clusters1, points, list_data[0])

clusters2, points = create_clusters(list_data[1], eps=0.1)
data2 = human_only(clusters2, points, list_data[1])

clusters3, points = create_clusters(list_data[2], eps=0.1)
data3 = human_only(clusters3, points, list_data[2])

clusters4, points = create_clusters(list_data[3], eps=0.2)
data4 = human_only(clusters4, points, list_data[3], largest_cluster_label=50)

clusters5, points = create_clusters(list_data[4], eps=0.1)
data5 = human_only(clusters5, points, list_data[4], largest_cluster_label=108)

clusters6, points = create_clusters(list_data[5], eps=0.2)
data6 = human_only(clusters6, points, list_data[5], largest_cluster_label=2)

# Combine datasets
combined_data = pd.concat([data1, data2, data4, data5, data6], ignore_index=True)

# Encode labels
label_encoder = LabelEncoder()
combined_data['Label'] = label_encoder.fit_transform(combined_data['Label'])

# Split data into features and labels
X = combined_data[['Points_X', 'Points_Y', 'Points_Z', 'Intensity']]
y = combined_data['Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Standardize features
# Convert data to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

# Define Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameters to tune
learning_rates = [0.0001, 0.001, 0.01]
num_layers_options = [2, 3, 4]
hidden_sizes = [64, 128, 256]

best_accuracy = 0.0
best_hyperparameters = {}

# Initialize lists to store training progress
train_losses = []
val_losses = []

# Iterate over hyperparameter combinations
for lr, num_layers, hidden_size in itertools.product(learning_rates, num_layers_options, hidden_sizes):
    print(f"Training with lr={lr}, num_layers={num_layers}, hidden_size={hidden_size}")
    accuracies = []
    for train_index, val_index in kf.split(X_tensor):
        X_train, X_val = X_tensor[train_index], X_tensor[val_index]
        y_train, y_val = y_tensor[train_index], y_tensor[val_index]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Instantiate model
        model = NeuralNetwork(input_size=X_train.shape[1], num_layers=num_layers, hidden_size=hidden_size).to(device)

        # Define Loss and Optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the Model
        model.train()
        epoch_train_losses = []
        for epoch in range(10):  # Adjust number of epochs as needed
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_train_losses.append(running_loss / len(train_loader))

        train_losses.append(epoch_train_losses)

        # Evaluate on Validation Set
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
                val_loss += criterion(outputs, labels.to(device)).item()
        val_losses.append(val_loss / len(val_loader))

        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Validation Accuracy: {accuracy:.4f}")

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"Avg. Validation Accuracy: {avg_accuracy:.4f}")

    # Update the best hyperparameters if this set performs better
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_hyperparameters = {'lr': lr, 'num_layers': num_layers, 'hidden_size': hidden_size}

    # Plotting training progress
    plt.figure(figsize=(10, 5))
    for i, epoch_losses in enumerate(train_losses):
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label=f'Fold {i + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss per Epoch (lr={lr}, num_layers={num_layers}, hidden_size={hidden_size})')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Fold')
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss per Fold (lr={lr}, num_layers={num_layers}, hidden_size={hidden_size})')
    plt.grid(True)
    plt.show()

print(f"Best Hyperparameters: {best_hyperparameters}")
print(f"Best Validation Accuracy: {best_accuracy:.4f}")

# import torch
# import torch.nn as nn
#
# Function to preprocess and predict using the trained model


