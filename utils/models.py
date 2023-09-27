import torch.nn as nn
import torch


class FCDN(nn.Module):
    """Fully Connected Deep Neural Network"""
    def __init__(self, input_shape, output_shape,activation='relu',device='cpu'):
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
        self.device = device
        super().__init__()
        
        self.model = nn.Sequential(
                nn.Linear(input_shape, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16,1)
            )
    
        self.model.to(device)

    def forward(self, x):
        #forward pass
        return self.model(x)
    
    
def test_fcdn():
    input_shape = 4
    output_shape = 1
    batch_size = 16
    model = FCDN(input_shape, output_shape)
    x = torch.rand(batch_size, input_shape )
    y = model(x)
    assert (y.shape[-1] == output_shape) , "output shape is not correct"


if __name__ == '__main__':
    test_fcdn()

