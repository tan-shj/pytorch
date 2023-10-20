import torch 
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self,filepath):
        data = np.genfromtxt("D:\github\pytorch\diabetes.csv",delimiter=',',dtype=np.float32)
        self.len = data.shape[0] 
        data = torch.from_numpy(data)
        self.x_data = data[:,:-1]
        self.y_data = data[:,[-1]]

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len
    
dataset = Dataset("D:\github\pytorch\diabetes.csv")
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=0)#num_workers=0 使用0个多线程

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,2)
        self.linear4 = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

def test():
    y_pred = model(dataset.x_data)
    y_pred_label = torch.where(y_pred>=0.5,torch.tensor([1,0]),torch.tensor([0,0]))
    acc = torch.eq(y_pred_label,dataset.y_data).sum().item()/dataset.y_data.size(0)
    print("test acc=" ,acc)

epoch_list = []
loss_list = []
for epoch in range(100):
    for i,data in enumerate(train_loader,0):
        #enumerate返回枚举对象，从第0个开始，能够获取索引 如(0, 序列[0]), (1, 序列[1]), (2, 序列[2])
        inputs,labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        print(epoch,i,loss.item())
        epoch_list.append(epoch)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# test()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()



