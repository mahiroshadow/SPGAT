import torch.cuda
from torch.optim import Adam
from module.model import SPGAT
from util.parser import arg
from data_loader import CustomDataset
from torch.utils.data import DataLoader


device="cuda:0" if torch.cuda.is_available() else "cpu"

def train():
    model=SPGAT()
    optim=Adam(model.parameters(),lr=arg.lr,weight_decay=arg.weight_decay)
    customDataset=CustomDataset()
    data_loader=DataLoader(customDataset,batch_size=arg.batch_size,shuffle=True)
    model.train()
    for epoch in range(arg.epoches):
        total_loss=0.0
        for users,items in data_loader:
            users,items=users.to(device),items.to(device)
            optim.step(users,items)
            loss=model()
            loss.backward()
            optim.zero_grad()
            total_loss+=loss.item()
        print(f"[epoch{epoch}]loss:{total_loss/len(data_loader)}")

if __name__=='__main__':
    pass


