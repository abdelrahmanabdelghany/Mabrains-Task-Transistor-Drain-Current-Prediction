import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class Transistor_dataset(Dataset):
    """Transistor_dataset class."""

    def __init__(self, csv_file):
        """
        Arguments:
            csv_file (string): Path to the csv file with features and labels.
        """
        assert os.path.isfile(csv_file), "File does not exist"
        assert os.path.splitext(csv_file)[1] == ".csv", "File is not a csv file"
        self.Data_Frame = pd.read_csv(csv_file)
        assert len(self.Data_Frame) > 0, "File is empty"


    def __len__(self):
        return len(self.Data_Frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.Data_Frame.iloc[idx, :-1].values)
        labels = torch.tensor(self.Data_Frame.iloc[idx, -1])

        sample = {'features': features, 'labels': labels}

        return sample
    
    
def test_dataset():
    dataset = Transistor_dataset(csv_file='MODIFIED_DATA.csv')
    sample = dataset[0]
    print("sample features shape: ", sample['features'].shape, "  ||  ","sample labels shape: ", sample['labels'].shape)
    print("sample features : ", sample['features']," ||  ", "sample labels : ", sample['labels'])
    assert (sample['features'].shape == (6,)), "Incorrect shape of features"
    assert (sample['labels'].shape == ()), "Incorrect shape of labels"
    assert (type(sample['features']) == torch.Tensor), "Incorrect type of features"
    assert (type(sample['labels']) == torch.Tensor), "Incorrect type of labels"
    

if __name__ == '__main__':
    test_dataset()
    print('All tests passed!')