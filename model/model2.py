import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from utils.utils_visualization import plot_classified_human
import random
import numpy as np
import torch.optim as optim
from utils.data_processing import list_data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import optuna
from utils.utils import create_clusters, human_only
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torchsummary import summary

# Use the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_channels, num_conv_layers, conv_out_channels, num_fc_layers, hidden_size):
        super(ImprovedNeuralNetwork, self).__init__()
        self.input_channels = input_channels
        self.num_conv_layers = num_conv_layers
        self.conv_out_channels = conv_out_channels
        self.num_fc_layers = num_fc_layers
        self.hidden_size = hidden_size

        # Initialize layers
        self.conv_layers = self._initialize_conv_layers()
        self.flattened_size = self._get_flattened_size()  # Initialize flattened_size
        self.fc_layers = self._initialize_fc_layers()

    def _initialize_conv_layers(self):
        # Initialize convolutional layers
        conv_layers = []
        in_channels = self.input_channels
        for _ in range(self.num_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels, self.conv_out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            in_channels = self.conv_out_channels
        return nn.Sequential(*conv_layers)

    def _get_flattened_size(self):
        # Create a dummy input tensor to calculate the flattened size
        dummy_input = torch.zeros(1, self.input_channels, 4)
        dummy_output = self.conv_layers(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        return flattened_size

    def _initialize_fc_layers(self):
        # Initialize fully connected layers
        fc_layers = []
        fc_layers.append(nn.Linear(self.flattened_size, self.hidden_size))
        fc_layers.append(nn.ReLU())
        for _ in range(self.num_fc_layers - 1):
            fc_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(self.hidden_size, 1))
        fc_layers.append(nn.Sigmoid())
        return nn.Sequential(*fc_layers)

    def forward(self, x):
        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x

    def _check_pooling_validity(self):
        # Check if pooling operation is valid based on current tensor size
        dummy_input = torch.zeros(1, self.input_channels, 4)
        output_after_conv = self.conv_layers(dummy_input)
        return output_after_conv.size(2) > 1


# Function to preprocess and predict using the trained model
def predict(new_data, model, scaler):
    new_data_scaled = scaler.transform(new_data[['Points_X', 'Points_Y', 'Points_Z', 'Intensity']])
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(new_data_tensor).squeeze().cpu().numpy()
    predicted_labels = ['Human' if pred >= 0.5 else 'No Human' for pred in predictions]
    new_data['Predicted_Label'] = predicted_labels
    return new_data


def objective(trial):
    # Hyperparameters to tune
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Hyperparameters to tune
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    conv_out_channels = trial.suggest_int('conv_out_channels', 16, 64, step=16)
    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 32, 128, step=32)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)

    # Model
    model = ImprovedNeuralNetwork(input_channels=1, num_conv_layers=num_conv_layers,
                                  conv_out_channels=conv_out_channels, num_fc_layers=num_fc_layers,
                                  hidden_size=hidden_size).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # KFold Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    accuracies = []

    for train_index, valid_index in kf.split(train_dataset):
        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        valid_subset = torch.utils.data.Subset(train_dataset, valid_index)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

        # Training loop
        model.train()
        for epoch in range(10):
            for batch in train_loader:
                X_batch, y_batch = batch
                optimizer.zero_grad()
                outputs = model(X_batch.unsqueeze(1).to(device))
                loss = criterion(outputs.squeeze(), y_batch.squeeze().to(device))
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch.unsqueeze(1).to(device))
                predicted_labels = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
                predictions.extend(predicted_labels.to(device).numpy())
                ground_truth.extend(y_batch.to(device).numpy())

        # Calculate accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        accuracies.append(accuracy)

    # Average accuracy across folds
    avg_accuracy = np.mean(accuracies)

    # Return average accuracy as the objective value
    return avg_accuracy

#################### Prepare Data For NN ####################

 # Create Label Dataset
clusters1, points = create_clusters(list_data[0], eps=0.1)
data1 = human_only(clusters1, points, list_data[0])

clusters2, points = create_clusters(list_data[1], eps=0.1)
data2 = human_only(clusters2, points, list_data[1])

# clusters3, points = create_clusters(list_data[2], eps=0.1)
# data3 = human_only(clusters3, points, list_data[2])

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

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)
test_dataset = TensorDataset(X_test, y_test)

# # Hyperparameter tuning with Optuna
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=10)
#
# # Best hyperparameters
# best_params = study.best_params
# print("Best hyperparameters: ", best_params)


best_hyperparams = {
    'num_conv_layers': 2,
    'conv_out_channels': 48,
    'num_fc_layers': 1,
    'hidden_size': 128,
    'learning_rate': 0.0007397851380541268,
    'batch_size': 64
}

with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_hyperparams, f)

with open('best_hyperparameters.json', 'r') as f:
    loaded_hyperparams = json.load(f)

# Training the final model with best hyperparameters
best_model = ImprovedNeuralNetwork(input_channels=1,
                                     num_conv_layers=loaded_hyperparams['num_conv_layers'],
                                     conv_out_channels=loaded_hyperparams['conv_out_channels'],
                                     num_fc_layers=loaded_hyperparams['num_fc_layers'],
                                     hidden_size=loaded_hyperparams['hidden_size']).to(device)
print(summary(best_model, (1, 4)))
optimizer = optim.Adam(best_model.parameters(), lr=loaded_hyperparams['learning_rate'])
criterion = nn.BCELoss()

# train_loader = DataLoader(train_dataset, batch_size=loaded_hyperparams['batch_size'], shuffle=True)
# #
# train_losses = []
# num_epochs = 12
# # Train the final model
# best_model.train()
# for epoch in range(num_epochs):  # Use more epochs for the final training
#     running_loss = 0.0
#     for batch in train_loader:
#         X_batch, y_batch = batch
#         optimizer.zero_grad()
#         outputs = best_model(X_batch.unsqueeze(1))
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     avg_loss = running_loss / len(train_loader)
#     train_losses.append(avg_loss)
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
# # Plot training loss
# plt.figure()
# plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
#
# # Validate the final model
# best_model.eval()
# valid_loss = 0
# correct = 0
# total = 0
# val_loader = DataLoader(valid_dataset, batch_size=loaded_hyperparams['batch_size'], shuffle=False)
# with torch.no_grad():
#     for X_batch, y_batch in val_loader:
#         outputs = best_model(X_batch.unsqueeze(1))
#         valid_loss += criterion(outputs, y_batch).item()
#         predicted = (outputs >= 0.5).float()
#         total += y_batch.size(0)
#         correct += (predicted == y_batch).sum().item()
#
# accuracy = correct / total
# print(f"Validation Accuracy: {accuracy:.4f}")
#

#
#     # # Save models weights
PATH = "C:\\Users\\Giota.x\\PycharmProjects\\LiDAR_Detection\\weights\\model2_weights.pt"
# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in best_model.state_dict():
#     print(param_tensor, "\t", best_model.state_dict()[param_tensor].size())
# #
# torch.save(best_model.state_dict(), PATH)
best_model.load_state_dict(torch.load(PATH))

# Testing the final model
best_model.eval()
test_loss = 0
correct = 0
total = 0
test_loader = DataLoader(test_dataset, batch_size=loaded_hyperparams['batch_size'], shuffle=False)

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = best_model(X_batch.unsqueeze(1))
        test_loss += criterion(outputs, y_batch).item()
        predicted = (outputs >= 0.5).float()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

pred_data3 = predict(list_data[0], best_model, scaler)
plot_classified_human(pred_data3, True)

# data3f = detect_floor(data3)
# plot_figure3D_floor(data3f)