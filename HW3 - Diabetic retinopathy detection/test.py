import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import *
import torchvision.models as models
from model import *
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


folder_name = 'ResNet50_pre_10e-5/'
# folder_name = './try/'
params_save_dir = './experiments/'+folder_name

if not os.path.isdir(params_save_dir):
    os.makedirs(params_save_dir)

# parameter
epoch_num = 4
batch_size = 4

# Load data
dataset_test = RetinopathyLoader('./data/', 'test')
loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False)

# Model Init
cuda = True if torch.cuda.is_available() else False
# print(cuda)
GPUID = [7]
torch.cuda.set_device(GPUID[0])

ResNet = ResNet50(False)
# ResNet = ResNet50(False)
ResNet.load_state_dict(torch.load('%sResNet_%d.pth' % (params_save_dir, epoch_num)))

ResNet.cuda(GPUID[0])

ResNet.eval()
acc = 0
num = 0
pred_list = []
label_list = []
with torch.no_grad():
    for step, (data, label) in enumerate(loader_test):
        result = ResNet(data.float().cuda(GPUID[0]))

        _, indices = torch.max(result, 1)
        for i in range(data.shape[0]):
            pred_list.append(indices.cpu().numpy()[i])
            label_list.append(label.cpu().numpy()[i])
        acc += torch.sum(torch.eq(indices.cuda(GPUID[0]), label.cuda(GPUID[0])).int()).cpu().numpy()
        num += data.shape[0]

acc /= num
print(acc)

def plot_cm(cm, classes, filename):

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

   
    for j, i in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, "{:.2f}".format(cm[j, i]), 
        color="white" if cm[j, i] > cm.max()/2. else "black", 
        horizontalalignment="center")


    plt.savefig(filename+'.png')
    plt.close(0)


cm = confusion_matrix(label_list, pred_list, normalize='true')
plot_cm(cm, 5, folder_name[:-1])

plt.savefig(folder_name[:-1]+'.png')
plt.close(0)
