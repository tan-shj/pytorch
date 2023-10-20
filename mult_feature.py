#多特征输入
import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.from_numpy(np.genfromtxt("D:\github\pytorch\diabetes.csv",delimiter=',',dtype=np.float32))
x_data = data[:,:-1]#此时x_data是2维
y_data = data[:,[-1]]#把y_data变成2维

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)#输入8维输出6维
        self.linear2 = torch.nn.Linear(6,4)#输入6维输出4维
        self.linear3 = torch.nn.Linear(4,2)#输入4维输出2维
        self.linear4 = torch.nn.Linear(2,1)#输入2维输出1维
        self.sigmoid = torch.nn.Sigmoid()#torch.nn中的模块，只需要输入，无参数
    def forward(self,x):
        y_pred = self.sigmoid(self.linear1(x))
        y_pred = self.sigmoid(self.linear2(y_pred))
        y_pred = self.sigmoid(self.linear3(y_pred))
        y_pred = self.sigmoid(self.linear4(y_pred))
        return y_pred
    
model = LogisticRegressionModel()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)


epoch_list = []
loss_list = []
for epoch in range(300):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_pred_label = torch.where(y_pred>=0.5,torch.tensor([1.0]),torch.tensor([0.0]))
# print(y_test.shape[0])  759
acc = torch.eq(y_pred_label, y_data).sum().item()/y_data.size(0)#准确率
print('acc=',acc)

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
