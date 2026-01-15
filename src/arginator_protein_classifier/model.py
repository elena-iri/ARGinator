from matplotlib.pylab import logistic
from torch import nn
import torch
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from torchmetrics import Accuracy

# class Model(nn.Module):
#     def __init__(self, input_dim, output_dim, dropout_rate)->None:
#         super().__init__()
#         self.input_dim = input_dim,
#         self.output_dim = output_dim,
#         self.fc1 = nn.Linear(1024, 256)
#         self.dropout=nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x
    
class Lightning_Model(LightningModule):
    def __init__(self, input_dim, output_dim, dropout_rate, loss_fn, optimizer)->None:
        super().__init__()
        self.save_hyperparameters() #save args for checkpointing
        #architecture
        self.input_dim = input_dim,
        self.output_dim = output_dim,
        self.fc1 = nn.Linear(1024, 256)
        self.dropout=nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # Loss & Optimizer are to change according to hydra
        self.loss_fn = instantiate(loss_fn) 
        self.optimizer = optimizer

        #Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc = Accuracy(task="multiclass", num_classes=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target)

        # Logging
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, params=self.parameters())
        return optimizer
    

if __name__ == "__main__":
    x = torch.rand(1, 1024)
    mock = Lightning_Model(x.shape[0], 2, 0.3)
    output = mock.forward(x)
    print(output.shape)