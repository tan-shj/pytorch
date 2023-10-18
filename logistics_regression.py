import torch
# import torchvision
import torch.nn.functional as F

# train_set = torchvision.datasets.MNIST(root='../dataset/mnist',train=True,download=True)
# test_set = torchvision.datasets.MNIST(root='../dataset/mnist',train=False,download=True)

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0],[0],[1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

#定义逻辑回归模型
model = LogisticRegressionModel()
#定义Loss函数(二分类loos)
criterion = torch.nn.BCELoss(reduction='sum')
#定义优化器(随机梯度下降)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)#计算预测值  调用Model中的forward
    loss = criterion(y_pred,y_data.float())#计算损失函数
    print(epoch,loss.data.item())

    optimizer.zero_grad()#优化器权值梯度初始化为0
    loss.backward()#反馈，计算偏导
    optimizer.step()#更新权值

#预测
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred=',y_test.data.item())
if y_test < 0.5:
    y_test = 0
else:
    y_test = 1
print('y_pred=',y_test)

