import torch
import matplotlib.pyplot as plt
import numpy as np

def transforms(sample):
    """
    Transforms sample.

    Arguments:
        sample: sample to be transformed

    Returns:
        transformed sample
    """
    #shift labels to avoid division by 0 in loss function
    sample['labels'] = sample['labels']+1
    
    return sample

def MAPE_loss(output, target):
    """
    Mean absolute percentage error loss.

    Arguments:
        output: output of the model
        target: target values

    Returns:
        MAPE loss.
    """
    return torch.mean(torch.abs((target - output) / (target)))  

def visualize_history_loss_acc(history, model_name="test",show=True,save=True):
    """
    Save Loss andd accuracy plots.

    Arguments: 
        history output of Trainer.train()
        model_name name of the model for saving the plots.
    
    Returns:
        None.
    """
    assert (list(history.keys())==["train_loss","val_loss","train_acc","val_acc"]) ,"history keys incorrect"
    assert (len(history['train_loss'])==len(history['val_loss'])==len(history['train_acc'])==len(history['val_acc'])),"history lengths incorrect"

    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    if save:
        plt.savefig(f'plots/{model_name}_loss.png')
    if show:
        plt.show()

    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    if save:
        plt.savefig(f'plots/{model_name}_acc.png')
    if show:
        plt.show()
    
    if save:
        print("Plots saved")

    

def visualize_data(y_pred,labels):
    """
    Visualizes data.

    Arguments:
        y_pred: predicted values
        labels: actual values

    Returns:
        None.
    """
    t=np.arange(0,y_pred.shape[0])
    plt.scatter(x=t,y=(y_pred-1).detach().cpu().numpy(), label='Predicted',alpha=0.5)
    plt.scatter(x=t,y=(labels-1).detach().cpu().numpy(), label='Actual',alpha=0.5)
    plt.legend()
    plt.show()
