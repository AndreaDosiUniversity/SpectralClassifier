import torch.nn as nn
import torch

class SpectraCNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SpectraCNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((input_size - 4) // 2) , 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#input_size = len(dataset[0][0])
#num_classes = len(np.unique(dataset.labels))
#
# model = SpectraCNNClassifier(input_size, num_classes)
