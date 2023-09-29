from dataset import Transistor_dataset
from models import FCDN
from train import Trainer
import torch
from dataset import Transistor_dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchmetrics.regression import MeanAbsolutePercentageError
from helpers import MAPE_loss
from helpers import visualize_history_loss_acc
import os
import torch.nn as nn
from helpers import transforms


def test_Transistor_dataset():
    """
    Test Transistor_dataset class.
    
    """
    dataset = Transistor_dataset(csv_file='MODIFIED_DATA.csv',transform=transforms)
    sample = dataset[0]
    assert (sample['features'].shape == (6,)), "Incorrect shape of features  " +f"shape= {sample['features'].shape}"
    assert (sample['labels'].shape == (1,)), "Incorrect shape of labels  " + f"shape= {sample['labels'].shape}"
    assert (type(sample['features']) == torch.Tensor), "Incorrect type of features  "+ f"type= {sample['features'].dtype}"
    assert (type(sample['labels']) == torch.Tensor), "Incorrect type of labels  "+ f"type={sample['labels'].dtype}"


def test_FCDN_Model():
    """
    Test FCDN model.

    """
    input_shape = 4
    output_shape = 1
    batch_size = 16
    model = FCDN(model_name='test_fcdn',input_shape=input_shape, output_shape=output_shape,activation=nn.ReLU(),device='cpu')
    x = torch.rand(batch_size, input_shape )
    y = model(x)
    assert (y.shape[-1] == output_shape) , "output shape is not correct  "+ f"shape= {y.shape}"
    model.save()
    assert (os.path.exists('saved_models/test_fcdn.pth')==True) , "model not saved"


def test_Trainer():
    """
    Test Trainer class.

    """
    input_shape = 6
    output_shape = 1
    batch_size = 256
    epochs = 1
    model = FCDN(model_name='test_fcdn',input_shape=input_shape, output_shape=output_shape,activation=nn.ReLU(),device='cuda')
    dataset=Transistor_dataset('Dataset_sample.csv',transform=transforms)
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size-valid_size

    train_dataset, valid_dataset ,test_dataset= random_split(dataset, [train_size, valid_size,test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

    #loss_fn = torch.nn.L1Loss()
    loss_fn = MAPE_loss
    accuracy = MeanAbsolutePercentageError().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer,accuracy=accuracy, device='cuda')
    history=trainer.train(train_loader=train_loader,val_loader=valid_loader,epochs=epochs)
    for batch in test_loader:
        Featurs,labels = batch['features'].to(device='cuda',dtype=torch.float),batch['labels'].to(device='cuda',dtype=torch.float)
        trainer.test(x=Featurs,y=labels)
        break
    assert (len(history['train_loss'])==epochs)
    model.save()
    assert (os.path.exists('saved_models/test_fcdn.pth')==True) , "model not saved"
    model.load(PATH='saved_models/test_fcdn.pth')
    assert (os.path.exists('saved_models/test_fcdn.pth')==True) , "model not loaded"


def test_save_plots():
    """
    Test save_plots function.
    """
    
    history={
    "train_loss":[],
    "val_loss":[],
    "train_acc":[],
    "val_acc":[]
    }
    history['train_loss']=[1,2,3,4,5]
    history['val_loss']=[1,2,3,4,5]
    history['train_acc']=[1,2,3,4,5]
    history['val_acc']=[1,2,3,4,5]
    history['val_acc']=[1,2,3,4,5]
    visualize_history_loss_acc(history,'test_fcdn',show=False,save=True)
    assert (os.path.exists('plots/test_fcdn_loss.png')==True) , "plots not saved"

