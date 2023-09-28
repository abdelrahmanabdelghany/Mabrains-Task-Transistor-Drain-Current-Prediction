import torch
def MAPE_loss(output, target):
    return torch.sum(torch.abs((target - output) / (target+1e-9)))  
