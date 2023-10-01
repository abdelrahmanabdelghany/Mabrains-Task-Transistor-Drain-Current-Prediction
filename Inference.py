from utils.arg_parser import parse_arguments
from utils.dataset import Transistor_dataset
from utils.models import FCDN
from utils.helpers import *
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd

def main():

    #get arguments
    args=parse_arguments()
    model_name=args.model_name
    dataset_name=args.dataset_name
    device=args.device


    # create dataset and dataloader
    test_dataset=Transistor_dataset(csv_file='data/'+dataset_name+'.csv',test_transform=test_datset_transforms)
    test_dataloader=DataLoader(test_dataset,len(test_dataset),shuffle=False)
    print('---------------dataset loaded---------------')
    print('test_size : ',len(test_dataset))

    #load model
    model=FCDN(model_name=model_name,input_shape=6,output_shape=1,activation=nn.LeakyReLU(),device=device)
    model.load(PATH='saved_models/'+model_name+'.pth')
    print('---------------model loaded---------------')


    #get predictions
    print('---------------predicting---------------')
    for batch in test_dataloader:
        Featurs,labels = batch['features'].to(device=device,dtype=torch.float),batch['labels'].to(device=device,dtype=torch.float)
        break
    model.eval()
    x = Featurs.to(device,dtype=torch.float)
    y = labels.to(device,dtype=torch.float)
    y_pred = model(x)
    

    # get accuracy
    accuracy=MAPE_loss(y_pred,y)
    print('---------------accuracy---------------')
    print('MAPE : ',accuracy)
    print('Marginal acc (0.1 margin) : ' ,marginal_acc(y_pred,y,margin=0.1).item() )
    print('Marginal acc (0.05 margin) : ' ,marginal_acc(y_pred,y,margin=0.05).item() )
    print('Marginal acc (0.01 margin) : ' ,marginal_acc(y_pred,y,margin=0.01).item() )

    # save predictions in csv format

    pd.DataFrame((10**y_pred-10).detach().cpu().numpy(),columns=['predictions']).to_csv('predictions/'+dataset_name+'.csv',index=False)
    print('---------------data saved---------------')


#main function
if __name__ == '__main__':
    main()
