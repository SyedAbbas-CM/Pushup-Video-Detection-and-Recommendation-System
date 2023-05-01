import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model2 import KeyPointsTransformer
from sklearn.metrics import confusion_matrix


data_folder = "D:/AIPROJECT2/output"

X_train = np.load(f"{data_folder}/X_train.npy")
y_train = np.load(f"{data_folder}/y_train.npy")
X_val = np.load(f"{data_folder}/X_val.npy")
y_val = np.load(f"{data_folder}/y_val.npy")
X_test = np.load(f"{data_folder}/X_test.npy")
y_test = np.load(f"{data_folder}/y_test.npy")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Set the sequence length (number of frames in a sequence)
sequence_length = 60

# Update input data shape
X_train = X_train.reshape(-1, sequence_length, 51)
X_val = X_val.reshape(-1, sequence_length, 51)
X_test = X_test.reshape(-1, sequence_length, 51)

num_classes = 6  # Exercise and Non-Exercise
model = KeyPointsTransformer(num_classes)

# Hyperparameters
learning_rate = 0.000004
num_epochs = 50
batch_size = 64

# Convert data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

        optimizer.zero_grad()
        outputs = model(inputs)
        print("Input shape:", inputs.shape)
        print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Outputs shape: {outputs.shape}")

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    val_predicted_labels = []
    val_true_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Store predicted and true labels
            val_predicted_labels.extend(predicted.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    val_cm = confusion_matrix(val_true_labels, val_predicted_labels)
    print("Validation Confusion Matrix:")
    print(val_cm)

    # Calculate validation accuracy
    correct = sum([pred == true for pred, true in zip(val_predicted_labels, val_true_labels)])
    val_accuracy = 100 * correct / len(val_true_labels)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Testing
    model.eval()
    correct = 0
    total = 0
    test_predicted_labels = []
    test_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Store predicted and true labels
            test_predicted_labels.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    test_cm = confusion_matrix(test_true_labels, test_predicted_labels)
    print("Test Confusion Matrix:")
    print(test_cm)

    # Calculate test accuracy
    correct = sum([pred == true for pred, true in zip(test_predicted_labels, test_true_labels)])
    test_accuracy = 100 * correct / len(test_true_labels)
    print(f"Test Accuracy: {test_accuracy:.2f}%")


    torch.save(model.state_dict(), "D:/AIPROJECT2/models/trained_model2.pth")
