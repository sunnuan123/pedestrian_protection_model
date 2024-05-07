'''
@File  :deeplearn_model.py
@Author:SunNuan
@Date  :2023/12/14 11:35
@Desc  :
'''
import random
from os.path import join
from torch.nn import init
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess_data import getData
import numpy as np
from tensorboardX import SummaryWriter

display = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_train, x_test, y_train, y_test = getData()


class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层: 全连接层
        self.fc1 = torch.nn.Linear(36, 128)
        # 第二层: 全连接层
        self.fc2 = torch.nn.Linear(128, 1)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)  # normal: mean=0, std=1


def item2tensor():
    # 数据标准化处理
    x_train_t = torch.from_numpy(x_train.values.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.values.astype(np.float32)).to(device)
    x_test_t = torch.from_numpy(x_test.values.astype(np.float32)).to(device)
    y_test_t = torch.from_numpy(y_test.values.astype(np.float32)).to(device)

    train_data = Data.TensorDataset(x_train_t, y_train_t)
    test_data = Data.TensorDataset(x_test_t, y_test_t)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=0)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=0)
    return train_loader, test_loader

train_loader, test_loader = item2tensor()


def get_metrics(predicts, labels):
    '''计算mae, r2测试指标'''
    # predicts = np.asarray(predicts)
    # labels = np.asarray(labels)
    mae = mean_absolute_error(labels, predicts)
    mse = mean_squared_error(labels, predicts)
    r2 = r2_score(labels, predicts)
    return mae, mse, r2

def validate(net):
    net.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for step,(x,y) in enumerate(test_loader):
            y_pred = net(x)
            y_pred = y_pred.data.cpu().numpy().squeeze().tolist()
            preds += y_pred
            labels += y.data.cpu().numpy().squeeze().tolist()
    mae, mse, r2 = get_metrics(preds, labels)
    return mae, mse, r2
def show(avg_loss,mse_test_loss,mae,r2,epoch):
    display.add_scalars('loss/train_test_loss',
                        {'train_loss': avg_loss,
                         'test_loss': mse_test_loss},
                        epoch)
    display.add_scalar('metric/mae', mae, epoch)
    display.add_scalar('metric/r2', r2, epoch)


def save_network(which_epoch, net):
    """save model to disk"""
    save_filename = 'epoch_%s_net.pth' % (which_epoch)
    save_path = join('./checkpoints', save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        torch.save(net.module.cpu().state_dict(), save_path)
        net.cuda('0')
    else:
        torch.save(net.cpu().state_dict(), save_path)


def train():
    # Construct our model by instantiating the class defined above
    net = DynamicNet().to(device)
    net.initialize()
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
    # 对模型迭代训练，总共epoch轮

    for epoch in range(500):
        net.train()
        avg_loss = []
        # 对训练数据的加载器进行迭代计算
        for x, y in train_loader:
            output = net(x)
            loss = criterion(output, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        avg_loss = np.array(avg_loss).mean()
        mae, mse_test_loss, r2 = validate(net)
        show(avg_loss, mse_test_loss, mae, r2, epoch)
        print("Epoch {}, train loss:{}, val loss:{}".format(epoch, avg_loss, mse_test_loss))
train()



























