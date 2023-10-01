import torch.nn as nn
import torch
import os


class FCDN(nn.Module):
    """Fully Connected Deep Neural Network"""
    def __init__(self,*,model_name, input_shape, dropout=0.2,output_shape,activation,device='cpu'):
        """
        Fully Connected Deep Neural Network
        
        Arguments:
            model_name: name of the model(used to save model later on)
            input_shape: input size
            output_shape: output size
            activation: activation function
            device: default cpu
        """
        super().__init__()

        assert (isinstance(input_shape, int)) , "input shape is not correct"
        assert (isinstance(output_shape, int)) , "output shape is not correct"
        assert (isinstance(activation, nn.Module)) , "activation is not correct"
        
        self.input = input_shape
        self.output = output_shape
        self.activation = activation
        self.device = device
        self.model_name = model_name
        self.dropout = nn.Dropout(dropout)
        
        
        self.model = nn.Sequential(
                nn.Linear(input_shape, 32),
                self.activation,
                nn.BatchNorm1d(32),
                nn.Linear(32, 32),
                self.activation,
                nn.BatchNorm1d(32),
                nn.Linear(32,32),
                self.activation,
                nn.BatchNorm1d(32),
                nn.Linear(32,32),
                self.activation,
                nn.BatchNorm1d(32),
                nn.Linear(32,32),
                self.activation,
                nn.BatchNorm1d(32),
                nn.Linear(32,output_shape)
            )
        
        self.init_weights()
        self.model.to(device)
    
    def _init_weights(self,layer):
        """
        Initialize weights.

        Arguments:
            layer: layer
        Returns:
            None
        """
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)

    def init_weights(self):
        """
        Initialize weights.

        Arguments:
            None
        Returns:
            None
        """
        self.model.apply(self._init_weights)




    def forward(self, x):
        #forward pass
        return self.model(x)
    
    def load(self,*,PATH):
        """
        Load model
        Arguments:
            PATH: model path to load model
        """
        assert os.path.isfile(PATH), "File does not exist"
        assert os.path.splitext(PATH)[1] == ".pth", "File is not a pth file"
        self.model.load_state_dict(torch.load(PATH))
        print(f"Model loaded from {PATH}")
    
    def save(self,PATH='saved_models/'):
        """
        Save model
        Arguments:
            PATH: folder path to save model
        """
        torch.save(self.model.state_dict(),  PATH+self.model_name+'.pth')
        assert os.path.isfile(PATH+self.model_name+'.pth'), "File does not exist"
        print(f"Model saved to {PATH+self.model_name}.pth")
