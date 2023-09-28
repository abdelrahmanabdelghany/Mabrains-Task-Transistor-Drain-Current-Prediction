import torch.nn as nn
import torch


class FCDN(nn.Module):
    """Fully Connected Deep Neural Network"""
    def __init__(self,*,model_name, input_shape, output_shape,activation=True,device='cpu'):
        """
        Arguments:
            input_shape: input size
            output_shape: output size
            activation: activation function
            device: default cpu
        """
        assert (isinstance(input_shape, int)) , "input shape is not correct"
        assert (isinstance(output_shape, int)) , "output shape is not correct"
        
        self.input = input_shape
        self.output = output_shape
        self.activation = activation
        self.device = device
        self.model_name = model_name
        
        super().__init__()
        
        self.model = nn.Sequential(
                nn.Linear(input_shape, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128,64),
                nn.Tanh(),
                nn.Linear(64,32),
                nn.Tanh(),
                nn.Linear(32,1)
            )
        
        if self.activation:
            self.model.add_module('activation',nn.ReLU())
    
        self.model.to(device)

    def forward(self, x):
        #forward pass
        return self.model(x)
    
    def save(self,PATH='saved_models/'):
        """
        Save model
        Arguments:
            PATH: folder path to save model
        """
        torch.save(self.model, PATH+self.model_name+'.pt')
