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
        Featurs=x.to(self.device,dtype=torch.float)
        Labels=y.to(self.device,dtype=torch.float)
        self.optimizer.zero_grad()
        self.model.train()
        y_pred = self.model(Featurs)
        loss = self.loss_fn(y_pred, Labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(),self.accuracy(y_pred,Labels)
    

    def _val_step(self, *, x, y):
        Featurs=x.to(self.device,dtype=torch.float)
        Labels=y.to(self.device,dtype=torch.float)
        self.model.eval()
        y_pred = self.model(Featurs)
        loss = self.loss_fn(y_pred, Labels)
        return loss.item(),self.accuracy(y_pred,Labels)

        
        
    
    def train(self,*,train_loader,val_loader,epochs):
        print("Training Started")
        num_train_batches=len(train_loader)
        num_val_batches=len(val_loader)
        history={
            "train_loss":[],
            "val_loss":[],
            "train_acc":[],
            "val_acc":[]
            }
        for epoch in range(1,epochs+1):
            Epoch_Train_Loss=0
            Epoch_Train_Acc=0
            Epoch_Val_Loss=0
            Epoch_Val_Acc=0
           
            t=tqdm(train_loader)
            i=0
            for batch in t:
                Featurs,labels = batch['features'],batch['labels']
                Train_Loss,Train_Acc = self._train_step(x=Featurs,y=labels)
                Epoch_Train_Loss+=Train_Loss
                Epoch_Train_Acc+=Train_Acc
                if(i%30==0):
                    t.set_description(f"epoch {epoch} batch_loss {Train_Loss:.2f} batch_acc {Train_Acc:.2f}")
                i+=1


            
            Epoch_Train_Loss=Epoch_Train_Loss/num_train_batches
            Epoch_Train_Acc=Epoch_Train_Acc.item()/num_train_batches

            history["train_loss"].append(Epoch_Train_Loss)
            history["train_acc"].append(Epoch_Train_Acc)

            print(f"Epoch {epoch} Train_Loss {Epoch_Train_Loss:.2f} Train_Accuracy {Epoch_Train_Acc:.2f}")

            for batch in tqdm(val_loader):
                Featurs,labels = batch['features'],batch['labels']
                Val_Loss,Val_Acc = self._val_step(x=Featurs,y=labels)
                Epoch_Val_Loss+=Val_Loss
                Epoch_Val_Acc+=Val_Acc
            
            Epoch_Val_Loss=Epoch_Val_Loss/num_val_batches
            Epoch_Val_Acc=Epoch_Val_Acc.item()/num_val_batches
            history["val_loss"].append(Epoch_Val_Loss)
            history["val_acc"].append(Epoch_Val_Acc)

            print(f"Epoch {epoch} Val_Loss {Epoch_Val_Loss:.2f} Val_Accuracy {Epoch_Val_Acc:.2f}")
           
        print("Training Completed")
        return history
    
    
    def  test(self,*,x,y):
        self.model.eval()
        x = x.to(self.device,dtype=torch.float)
        y = y.to(self.device,dtype=torch.float)
        y_pred = self.model(x)
        Test_Acc=self.accuracy(y_pred,y)
        print(f"Test_Accuracy {Test_Acc:.2f}")
        return Test_Acc




