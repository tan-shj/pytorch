import torch

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1,bias=True)
    #计算预测值
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
#定义线性模型
model = LinearModel()
#定义Loss函数(均值平均误差)
criterion = torch.nn.MSELoss(reduction='sum')
#定义优化器(随机梯度下降)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)#计算预测值  调用LinearModel中的forward
    loss = criterion(y_pred,y_data)#计算损失函数
    print(epoch,loss.data.item())

    optimizer.zero_grad()#优化器权值梯度初始化为0
    loss.backward()#反馈，计算偏导
    optimizer.step()#更新权值

#item()取出权值的标量
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

#预测
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred=',y_test.data.item())
