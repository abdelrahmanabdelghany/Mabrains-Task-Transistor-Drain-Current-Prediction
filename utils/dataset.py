import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class Transistor_dataset(Dataset):
    """Transistor_dataset class."""

    def __init__(self, csv_file,test_transform=None,train_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with features and labels.
            test_transform (callable, optional): Optional transform to be applied
                on a sample for test data.
            train_transform (callable, optional): Optional transform to be applied
                on a sample for train data.
        
        """
        assert os.path.isfile(csv_file), "File does not exist"
        assert os.path.splitext(csv_file)[1] == ".csv", "File is not a csv file"
        self.Data_Frame = pd.read_csv(csv_file)
        assert len(self.Data_Frame) > 0, "File is empty"
        self.train_transform = train_transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.Data_Frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.Data_Frame.iloc[idx, :-1].values)
        labels = torch.tensor([self.Data_Frame.iloc[idx, -1]])

        sample = {'features': features, 'labels': labels}

        if self.test_transform:
            sample = self.test_transform(sample)
        
        if self.train_transform:
            sample = self.train_transform(sample)

        return sample

    
    
