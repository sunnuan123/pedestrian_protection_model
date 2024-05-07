'''
@File  :deeplearn_model2.py
@Author:SunNuan
@Date  :2023/12/17 11:48
@Desc  :
'''
from os.path import join

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from torch import nn



import os
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


train_data, test_data = train_test_split(pd.read_csv('./data/stdData.csv'),
                                         test_size=0.1, random_state=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, raw_data, transform=None, target_transform=None):
        self.raw_data = raw_data.values
        self.data = self.raw_data[:,:-1]
        self.target = self.raw_data[:,-1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.target[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


train_data = CustomDataset(train_data)
test_data = CustomDataset(test_data)

train_dataloader = DataLoader(train_data,
                              batch_size=16,
                              shuffle=True,
                              num_workers=0)

test_dataloader = DataLoader(test_data,
                              batch_size=16,
                              shuffle=False,
                              num_workers=0)


class NetModel(nn.Module):
    def __init__(self, out_node):
        super(NetModel, self).__init__()
        self.linears = [36, 48, 16, 8, 1]
        for i in range(len(self.linears)-1):
            setattr(self, 'linear{}'.format(i), nn.Linear(self.linears[i], self.linears[i + 1]))
            nn.init.xavier_uniform_(eval('self.linear{}'.format(i)).weight)
            nn.init.zeros_(eval('self.linear{}'.format(i)).bias)
        # self.hid1 = nn.Linear(36, 48)
        # nn.init.xavier_uniform_(self.hid1.weight)
        # nn.init.zeros_(self.hid1.bias)
        # self.hid2 = nn.Linear(48,16)
        # nn.init.xavier_uniform_(self.hid2.weight)
        # nn.init.zeros_(self.hid2.bias)
        # self.oupt = nn.Linear(16, out_node)
        # nn.init.xavier_uniform_(self.oupt.weight)
        # nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        x = x.to(torch.float32)
        for i in range(len(self.linears) - 1):
            x = getattr(self, 'linear{}'.format(i))(x)
            # x = F.relu(getattr(self, 'norm{}'.format(i))(x))
        # x = x.to(torch.float32)
        # # z = torch.tanh(self.hid1(x))
        # x = self.hid1(x)
        # # z = torch.tanh(self.hid2(z))
        # x = self.hid2(x)
        # x = self.oupt(x)
        return x

class Writer():
    def __init__(self):
        self.display = SummaryWriter()

    # 可视化预测结果和真是结果
    def plotOutputs(self, y_test, y_pred, alg_name, r2):
        # Plot outputs
        plt.figure(figsize=(12, 6))
        x = range(len(y_test))
        l1 = plt.plot(x, y_test, 'o-', color="red", linewidth=1, linestyle='-', label='y_true')
        # 使用一个列表（temp）来存储每个点的坐标
        y_test_value = [list(y_test)[i] for i in range(len(x))]
        # 有多少个点就循环多少次
        for i in range(len(x)):
            plt.annotate(y_test_value[i], xy=(x[i], y_test_value[i]),
                         xytext=(x[i] + 0.01, y_test_value[i] + 0.01), color='red')  # 这里xy是需要标记的坐标，xytext是对应的标签坐标
        l2 = plt.plot(x, y_pred, 'o-', color="blue", linewidth=1, linestyle='-.', label='y_pred')
        for i in range(len(x)):
            plt.annotate(np.round(y_pred[i], 4), xy=(x[i], y_pred[i]),
                         xytext=(x[i] - 0.1, y_pred[i] - 0.1), color='blue')  # 这里xy是需要标记的坐标，xytext是对应的标签坐标
        plt.xlabel('预测点的索引')
        plt.ylabel('y值')
        plt.xticks(x)
        plt.yticks(range(int(min(y_test)), int(max(y_test)), 80))
        plt.legend()
        plt.title(f'{alg_name}测试集上预测值和真实值对比图,test_r2={r2}')
        plt.show()
    def show(self, avg_loss, mse_test_loss, mae, r2, epoch):
        self.display.add_scalars('loss/train_test_loss',
                            {'train_loss': avg_loss,
                             'test_loss': mse_test_loss},
                            epoch)
        self.display.add_scalar('metric/mae', mae, epoch)
        self.display.add_scalar('metric/r2', r2, epoch)


class TrainHelper():
    def __init__(self,train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.net = NetModel(1)# 建立网络
        print(self.net)
        # 这里也可以使用其它的优化方法
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00004)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.1)
        # 定义一个误差计算方法
        self.loss_func = torch.nn.MSELoss()
        self.writer = Writer()
    def train(self):
        total_batch = 0
        for epoch in range(1, 4000):
            for i, (data, label) in enumerate(train_dataloader, start=1):
                total_batch += i
                # 输入数据进行预测
                prediction = self.net(data)
                # 第一个参数为预测值，第二个为真值
                train_loss = self.loss_func(prediction.float(), torch.unsqueeze(label.float(),1))
                # 每次开始优化前将梯度置为0
                self.optimizer.zero_grad()
                # 误差反向传播
                train_loss.backward()
                # 按照最小loss优化参数
                self.optimizer.step()
                self.writer.display.add_scalar('train_loss', train_loss, total_batch)
                # print(f'train_loss/mse:{train_loss},total_batch:{total_batch},')
            # 验证
            test_mae, test_loss, test_r2, result, preds, labels = self.validate()
            self.writer.show(train_loss, test_loss, test_mae, test_r2, epoch)
            self.scheduler.step()
            if epoch % 10 == 0:
                print(f'test_mae：{test_mae}, test_loss：{test_loss}, test_r2：{test_r2}')
                print(result.head(10))
                self.save_network(epoch, self.net)
        self.writer.plotOutputs(labels, preds, 'neural_network', test_r2)

    def get_metrics(self, predicts, labels):
        '''计算mae, r2测试指标'''
        # predicts = np.asarray(predicts)
        # labels = np.asarray(labels)
        mae = mean_absolute_error(labels, predicts)
        mse = mean_squared_error(labels, predicts)
        r2 = r2_score(labels, predicts)
        return mae, mse, r2

    def validate(self):
        self.net.eval()
        preds = []
        labels = []
        result = pd.DataFrame()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_dataloader):
                y_pred = self.net(x)
                y_pred = y_pred.data.cpu().numpy().squeeze().tolist()
                preds += y_pred
                labels += y.data.cpu().numpy().squeeze().tolist()
        mae, mse, r2 = self.get_metrics(preds, labels)
        result['preds'] = preds
        result['labels'] = labels
        return mae, mse, r2,result, preds, labels

    def save_network(self, which_epoch, net):
        """save model to disk"""
        save_filename = 'epoch_%s_net.pth' % (which_epoch)
        save_path = join('./checkpoints', save_filename)
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(net.cpu().state_dict(), save_path)

if __name__=='__main__':
    th = TrainHelper(train_dataloader, test_dataloader)
    th.train()





