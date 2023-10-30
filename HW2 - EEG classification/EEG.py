import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import *
from model_ELU import *
# from model_ReLU import *
# from model_LeakyReLU import *
import os

folder_name = 'EEG_ELU_orthogonal0.01_10e-4_d50_100_e10000/'
params_save_dir = './experiments/'+folder_name
log_dir = './logs/'+folder_name

if not os.path.isdir(params_save_dir):
    os.makedirs(params_save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

# parameter
batch_size = 64
epoch_num = 10000
lr=1e-4

# Load data
dataset = SignalLoader('train')
loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
dataset_test = SignalLoader('test')
loader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False)

# Model Init
cuda = True if torch.cuda.is_available() else False
GPUID = [0]
torch.cuda.set_device(GPUID[0])
EEG = init_weights(EEGNet(), init_type='orthogonal', init_gain=0.01)
EEG.cuda(GPUID[0])

optimizer_EEG = torch.optim.Adam(EEG.parameters(), lr=lr)
torch.save(EEG.state_dict(), '%sEEG_init.pth'%(params_save_dir))

# LOSS
entroy=nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir)
acc_train = []
acc_test = []
for epoch in range(epoch_num):
    print("------epoch%d------"%(epoch))
    loss_avg = 0
    wrong = 0
    num = 0
    EEG.train()
    for step, (data, label) in enumerate(loader):

        result = EEG(data.float().cuda(GPUID[0]))

        # Loss
        loss = entroy(result.float().cuda(GPUID[0]), label.long().cuda(GPUID[0]))

        # Back propagation
        optimizer_EEG.zero_grad()
        loss.backward(retain_graph=True)

        optimizer_EEG.step()

        loss_avg += loss

        # Prediction
        _, indices = torch.max(result, 1)
        
        wrong += torch.sum(torch.abs(indices.cuda(GPUID[0]) - label.cuda(GPUID[0])))
        num += data.shape[0]

    # Record loss and accuracy
    loss_avg /= step
    wrong /= num
    acc_train.append(1-wrong)
    test_result = test(EEG, loader_test, GPUID, batch_size)
    acc_test.append(test_result)
    print("loss:%f"%(loss))
    print("acc_train:%f"%(1-wrong))
    print("acc_test:%f"%(test_result))

    writer.add_scalar('/loss/loss_avg', loss, epoch)
    writer.add_scalar('/accuracy/train', 1-wrong, epoch)
    writer.add_scalar('/accuracy/test', test_result, epoch)
    if (epoch+1)%50==0:
        torch.save(EEG.state_dict(), '%sEEG_%d.pth'%(params_save_dir, epoch))
    
    if epoch < 250 and (epoch+1)%50==0:
        lr /= (10**(1/5))
    elif (epoch+1)%100==0:
        lr /= (10**(1/5))

torch.save(acc_train, '%saccuracy_train.pth'%(params_save_dir))
torch.save(acc_test, '%saccuracy_test.pth'%(params_save_dir))
writer.close()
