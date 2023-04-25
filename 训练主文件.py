from load_dataset import load_data
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from UNet import UNet
from conbine_loss import CombinedLoss
import torch.optim as optim
import torch.nn as nn
path_test = "test_npz_seg_512x512_s1"
path_train ='train_npz_seg_512x512_s1'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np_test,np_test_mask = load_data(path_test)
np_train,np_train_mask = load_data(path_train)
# print(np_test.shape,np_test_mask.shape)
# print(np_train.shape,np_train_mask.shape)
# 归一化
test_data = (np_test/255).astype(np.float32)
test_mask = np_test_mask
train_data = (np_train/255).astype(np.float32)
train_mask = np_train_mask
train_data =torch.from_numpy(train_data)
train_data = train_data.permute(0, 3, 1, 2)
test_data = torch.from_numpy(test_data)
test_data = test_data.permute(0, 3, 1, 2)
## print(train_data.shape)
## print(train_mask.shape)
test_set = TensorDataset(test_data, torch.from_numpy(test_mask))
train_set = TensorDataset(train_data, torch.from_numpy(train_mask))

net = UNet(n_channels=3,n_classes=26).to(device)
criterion = nn.CrossEntropyLoss()#定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.010)  # 定义优化器
trainLoader = DataLoader(train_set, batch_size=1,
                         shuffle=True, num_workers=0)
testLoader = DataLoader(test_set, batch_size=1,
                         shuffle=False, num_workers=0)
import time
start = time.time()
for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data  # 获取数据
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(inputs.shape)
        optimizer.zero_grad()  # 清空梯度缓存
        outputs = net(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels.long())
        loss.backward()  # 反向传播
        optimizer.step()  # 调整模型
        running_loss += loss.item()
        if i % 10 == 9:
            # 每 20 次迭代打印一次信息
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss))
            running_loss = 0.0
print('Finish Traning! Total cost time: ', time.time()-start)



