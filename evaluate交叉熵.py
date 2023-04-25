from load_dataset import load_data
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from UNet import UNet
from conbine_loss import CombinedLoss
import torch.optim as optim
import torch.nn as nn

path_test = "test_npz_seg_512x512_s1"
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np_test,np_test_mask = load_data(path_test)
test_mask = np_test_mask
test_data = (np_test/255).astype(np.float32)
test_data = torch.from_numpy(test_data)
test_data = test_data.permute(0, 3, 1, 2)
test_set = TensorDataset(test_data, torch.from_numpy(test_mask))
net = torch.load('net.pkl', map_location=torch.device('cpu'))
testLoader = DataLoader(test_set, batch_size=1,
                         shuffle=False, num_workers=0)
net.eval()
true_positive_arr= np.zeros(26)
false_negative_arr = np.zeros(26)
false_positive_arr = np.zeros(26)
accuracy = []
output_np = []
with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        # print(labels.shape)
        outputs = net(inputs)
        output_np.append(outputs)
        # print(outputs.shape)
        # print(outputs[0,0:25,0,0])
        _,pred = torch.max(outputs, dim=1)
        # print(x_max_indices[0,0,0])
        correct = (pred == labels).sum().item()
        total = labels.numel()
        pixel_accuracy = correct / total
        accuracy.append(pixel_accuracy)
        for c in range(26):
            true_positive = ((pred == c) & (labels == c)).sum().item()
            false_negative = ((pred != c) & (labels == c)).sum().item()
            false_positive = ((pred == c) & (labels != c)).sum().item()
            true_positive_arr[c]+= true_positive
            false_negative_arr[c]+=false_negative
            false_positive_arr[c]+=false_positive
accuracy = np.array(accuracy)
output_np = np.array(output_np)
print(f'accuracy_mean:{np.mean(accuracy)}')
recall_per_class = true_positive_arr / (true_positive_arr + false_negative_arr)
IoU = true_positive_arr/(true_positive_arr+false_negative_arr+false_positive_arr)
print(f'recall{recall_per_class}')
print(f'mIoU{np.mean(IoU)}')
        