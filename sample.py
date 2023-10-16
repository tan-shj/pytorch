import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import ipdb

#print(torch.cuda.is_available())
print(torch.cuda.device_count())

n_samples = 1000
x = torch.rand(n_samples,10)
y = 3 * x.sum(dim=1) + 2 + torch.randn(n_samples)

dataset = TensorDataset(x,y)
dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP,self).__init__()
        self.fc1 = nn.Linear(10,5)
        self.fc2 = nn.Linear(5,1)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10):
    for batch_x,batch_y in dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(),batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch [{epoch+1}/10] , Loss:{loss.item():.4f}")
print("Training complete")

###ipdb.set_trace()   设置断点
###pyhton -m ipdb -c contiue sample.py  到第一个断点或报错点停止

print(f"Model is on:{next(model.parameters()).device}")

print("###############   GPU  ################")

device = torch.device("cuda")
print(device)

model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

model = model.to(device)

for epoch in range(10):
    for batch_x,batch_y in dataloader:
        #前向传播
        #ipdb.set_trace()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(),batch_y)

        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch [{epoch+1}/10] , Loss:{loss.item():.4f}")
print("Training complete")
print(f"Model is on:{next(model.parameters()).device}")
