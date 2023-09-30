import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from utils.helpers import test_datset_transforms

class Transistor_dataset(Dataset):
    """Transistor_dataset class."""

    def __init__(self, csv_file,test_data=False,transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with features and labels.
            test_data (bool, optional): If True, apply test_transform
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        """
        assert os.path.isfile(csv_file), "File does not exist"
        assert os.path.splitext(csv_file)[1] == ".csv", "File is not a csv file"
        self.Data_Frame = pd.read_csv(csv_file)
        assert len(self.Data_Frame) > 0, "File is empty"
        self.transform = transform
        self.test_data = test_data

    def __len__(self):
        return len(self.Data_Frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.Data_Frame.iloc[idx, :-1].values)
        labels = torch.tensor([self.Data_Frame.iloc[idx, -1]])

        sample = {'features': features, 'labels': labels}
        if self.test_data:
            sample = test_datset_transforms(sample)
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    
    
