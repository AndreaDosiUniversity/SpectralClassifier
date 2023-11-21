import torch.nn as nn
import torch

# class SpectralCNNClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SpectralCNNClassifier, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding = 2)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.fc1 = nn.Linear(64 * ((input_size - 4) // 2) , 128)
#         self.fc2 = nn.Linear(128, num_classes)
    
#     def forward(self, x):
#         x = x.unsqueeze(1)  # Add a channel dimension
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Flatten
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class SpectralCNNClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SpectralCNNClassifier, self).__init__()
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.fc1 = nn.Linear(64 * (input_size // 4), 1000)  # Adjusted based on input size
#         self.fc2 = nn.Linear(1000, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(-1, self.num_flat_features(x))
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # All dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
    


class SpectralCNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SpectralCNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512 * (input_size // 16), 512)  # Adjusted based on input size
        self.batch_norm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# # Assuming your input size is 19967 and you have 10 classes
# input_size = 19967
# num_classes = 10

# # Instantiate the model
# model = SpectraCNN(input_size, num_classes)

# # You can use CrossEntropyLoss for classification problems and Adam optimizer, for example
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Convert your dataset and labels to PyTorch tensors
# # Assuming your spectra data is in the variable 'spectra_data' and labels in 'labels'
# spectra_data = torch.Tensor(spectra_data)
# labels = torch.LongTensor(labels)

# # Training loop example
# num_epochs = 10
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(spectra_data.unsqueeze(1))  # Add a channel dimension (1 channel for grayscale)
#     loss = criterion(outputs, labels)

#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # After training, you can use the model for predictions
# # For example, if you have a new spectrum 'new_spectrum':
# new_spectrum = torch.Tensor(new_spectrum)
# output = model(new_spectrum.unsqueeze(0).unsqueeze(1))  # Add batch and channel dimensions
# predicted_class = torch.argmax(output).item()

# print(f'Predicted class: {predicted_class}')



# This model has two convolutional layers, each followed by a ReLU activation function and a max pooling layer. The output of the second pooling layer is flattened and passed through two fully connected layers to produce the final output.

# 1D CNN
# class SpectralCNNClassifier(nn.Module):
#     def __init__(self):
#         super(SpectralCNNClassifier, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=3)
#         self.fc1 = nn.Linear(16 * 9984, 64)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         print(x.shape)  
#         x = self.relu1(x)
#         print(x.shape)
#         x = self.pool1(x)
#         print(x.shape)
#         x = self.conv2(x)
#         print(x.shape)
#         x = self.relu2(x)
#         print(x.shape)
#         x = self.pool2(x)
#         print(x.shape)
#         x = x.view(-1, 32 * 9984)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.fc2(x)
#         return x

