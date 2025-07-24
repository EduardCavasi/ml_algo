import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.2)
        self.batch = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def show_data(x, y):
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i], cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_wrong_prediction(x, y, y_pred):
    plt.figure(figsize=(10, 4))
    k = 1
    for i in range(10000):
        if y_pred[i] != y[i]:
            plt.subplot(10, 10, k)
            k = k + 1
            plt.imshow(x[i], cmap='gray')
    plt.tight_layout()
    plt.show()

def transform_to_dataset(x, y):
    images_tensor = torch.from_numpy(x).float()
    labels_tensor = torch.from_numpy(y).long()
    images_tensor = images_tensor.permute(0, 3, 1, 2)  # (60000, 1, 28, 28)
    dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
    return dataset

data_path = "./data/MNIST/raw/"
x_train = load_images(os.path.join(data_path, "train-images.idx3-ubyte"))
y_train = load_labels(os.path.join(data_path, "train-labels.idx1-ubyte"))
x_test = load_images(os.path.join(data_path, "t10k-images.idx3-ubyte"))
y_test = load_labels(os.path.join(data_path, "t10k-labels.idx1-ubyte"))

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
#show_data()

train_ds = transform_to_dataset(x_train, y_train)
test_ds = transform_to_dataset(x_test, y_test)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=True)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

#training
epoch = 10
for ep in range(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:
            print(f'[{ep + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

with torch.no_grad():
    total = 0
    correct = 0
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total}")


