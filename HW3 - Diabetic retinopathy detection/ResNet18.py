import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import *
import torchvision.models as models
from model import *
import os
import numpy as np


folder_name = 'ResNet18_pre_10e-4_decay_conti_15to25/'
params_save_dir = './experiments/'+folder_name
log_dir = './logs/'+folder_name

if not os.path.isdir(params_save_dir):
    os.makedirs(params_save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)


# parameter
batch_size = 4
lr = 10e-7
epoch_begin = 15
epoch_num = 25
decay = 5e-4
is_pretrained = True

# Load data
dataset = RetinopathyLoader('./data/', 'train')
loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
dataset_test = RetinopathyLoader('./data/', 'test')
loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False)

# Model Init
cuda = True if torch.cuda.is_available() else False
# print(cuda)
GPUID = [7]
torch.cuda.set_device(GPUID[0])
if is_pretrained:
    ResNet = ResNet18(True)
else:
    ResNet = init_weights(ResNet18(False), init_type='xavier', init_gain=0.01)

ResNet.cuda(GPUID[0])

optimizer_ResNet = torch.optim.SGD(ResNet.parameters(), lr=lr, momentum=0.9)
torch.save(ResNet.state_dict(), '%sResNet_init.pth'%(params_save_dir))

# LOSS
entroy=nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir)
acc_train = []
acc_test = []
for epoch in range(epoch_begin, epoch_num):
    print("------epoch%d------"%(epoch))
    loss_avg = 0
    acc = 0
    num = 0
    ResNet.train()
    for step, (data, label) in enumerate(loader):
        result = ResNet(data.cuda(GPUID[0]))
        # print(data.shape)
        # Loss
        loss = entroy(result.cuda(GPUID[0]), label.long().cuda(GPUID[0]))

        # Back propagation
        optimizer_ResNet.zero_grad()
        loss.backward()

        optimizer_ResNet.step()

        loss_avg += loss

        # Prediction
        _, indices = torch.max(result, 1)
        
        
        acc += torch.sum(torch.eq(indices.cuda(GPUID[0]), label.cuda(GPUID[0])).int()).cpu().numpy()
        num += data.shape[0]
        
    lr /= (10**(1/5))

    # Record loss and accuracy
    loss_avg /= step
    
    acc /= num
    acc_train.append(acc)
    test_result = test(ResNet, loader_test, GPUID, batch_size)
    acc_test.append(test_result)
    print(indices, label)
    print("loss:%f"%(loss))
    print("acc_train:%f"%(acc))
    print("acc_test:%f"%(test_result))

    writer.add_scalar('/loss/loss_avg', loss, epoch)
    writer.add_scalar('/accuracy/train', acc, epoch)
    writer.add_scalar('/accuracy/test', test_result, epoch)

    torch.save(ResNet.state_dict(), '%sResNet_%d.pth'%(params_save_dir, epoch))



np.save('%saccuracy_train'%(params_save_dir), acc_train)
np.save('%saccuracy_test'%(params_save_dir), acc_test)
writer.close()
