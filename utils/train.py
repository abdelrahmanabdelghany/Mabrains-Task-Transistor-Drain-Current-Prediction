import torch
from tqdm import tqdm

class Trainer():
    """Class to handle training of a model."""

    def __init__(self,*, model, loss_fn, optimizer,accuracy, device):
        """
        Arguments:
            model: model to be trained
            loss_fn: loss function
            optimizer: optimizer
            accuracy: accuracy metric
            device: device on which to train the model, e.g. 'cpu' or 'cuda'
        """

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.accuracy=accuracy
        self.device = device

        


    def  _train_step(self, *, x, y):
        Featurs=x.to(self.device)
        Labels=y.to(self.device)
        self.optimizer.zero_grad()
        self.model.train()
        y_pred = self.model(Featurs)
        loss = self.loss_fn(y_pred, Labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(),self.accuracy(y_pred,Labels)
    

    def _val_step(self, *, x, y):
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        return loss.item(),self.accuracy(y_pred,y)

        
        
    
    def train(self,*,train_loader,val_loader,epochs):
        print("Training Started")
        num_train_batches=len(train_loader)
        num_val_batches=len(val_loader)
        history={
            "train_loss":[],
            "val_loss":[]
        }
        for epoch in range(1,epochs+1):
            Epoch_Train_Loss=0
            Epoch_Train_Acc=0
            Epoch_Val_Loss=0
            Epoch_Val_Acc=0
           
           
            for Featurs, label in tqdm(train_loader):
                Train_Loss,Train_Acc = self._train_step(x=Featurs,y=label)
                Epoch_Train_Loss+=Train_Loss
                Epoch_Train_Acc+=Train_Acc

            
            

            history["train_loss"].append(Epoch_Train_Loss/num_train_batches)
            history["train_acc"].append(Epoch_Train_Acc.item()/num_train_batches)

            print(f"Epoch {epoch} Train_Loss {Train_Loss} Train_Accuracy {Train_Acc}")

            for Featurs, label in tqdm(val_loader):
                Val_Loss,Val_Acc = self._val_step(x=Featurs,y=label)
                Epoch_Val_Loss+=Val_Loss
                Epoch_Val_Acc+=Val_Acc
                
            history["val_loss"].append(Epoch_Val_Loss/num_val_batches)
            history["val_acc"].append(Epoch_Val_Acc.item()/num_val_batches)

            print(f"Epoch {epoch} Val_Loss {Val_Loss} Val_Accuracy {Val_Acc}")
           
        print("Training Completed")
        return history
    
    
    def  test(self, x, y):
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        Test_Acc=self.accuracy(y_pred,y)
        print(f"Test_Accuracy {Test_Acc}")
        return Test_Acc


def test_training():
    from models import FCDN
    input_shape = 4
    output_shape = 1
    batch_size = 16
    model = FCDN(input_shape, output_shape)



if __name__ == '__main__':
    test_fcdn()