import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd


class SpectralDataSets(Dataset):
    def __int__ (self, csv_training_spectra, csv_training_labels, transform=None):
        training_spectra=pd.read_csv(csv_training_spectra)
        training_labels=pd.read_csv(csv_training_labels)
        self.spectra = self.training_spectra.iloc.values  # Extract spectra as numpy array
        self.labels = self.training_labels.iloc.values  # Extract labels as numpy array
        self.transform = transform

    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        label = self.labels[idx]
        #sample = {'spectrum': spectrum, 'label': label}

        if self.transform:
            spectrum = self.transform(spectrum)
        
        return spectrum, label
    
    from torch.utils.data import DataLoader

#batch_size = 32
#dataset = SpectralDataSets(csv_training_spectra, csv_training_labels)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

