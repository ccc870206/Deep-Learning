import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from dataloader import *
from torchvision.utils import save_image
from model import *
import time
import math


folder_name = 'main/'
params_save_dir = './experiments/'+folder_name
img_save_dir = './experiments/' + folder_name + 'img/'
log_dir = './logs/'+folder_name

if not os.path.isdir(params_save_dir):
    os.makedirs(params_save_dir)
if not os.path.isdir(img_save_dir):
    os.makedirs(img_save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

# parameter
img_size = (64, 64)
batch_size = 16
lr = 0.0002
epoch_num = 1000
decay = 5e-4
inputSize = 48
hiddenSize = 64
outputSize = 3
beta1 = 0.5

# Load data
dataset = ImageLoader('./clevr/', 'train', img_size)
loader = DataLoader(dataset, batch_size=batch_size,shuffle=True,drop_last = True)


# Model Init
cuda = True if torch.cuda.is_available() else False
GPUID = [1]
torch.cuda.set_device(GPUID[0])
G = Generator(inputSize, hiddenSize, outputSize)
D = Discriminator(3, hiddenSize)
G.apply(weights_init)
D.apply(weights_init)

G.cuda(GPUID[0])
D.cuda(GPUID[0])
unnormal = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],std = [ 1., 1., 1. ])])
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
torch.save(G.state_dict(), '%sG_init.pth'%(params_save_dir))
torch.save(D.state_dict(), '%sD_init.pth'%(params_save_dir))
optimizerD =  torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=10e-4)
optimizerG =  torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=10e-4)

writer = SummaryWriter(log_dir)
criterion = nn.BCELoss()
start = time.time()

for epoch in range(epoch_num):
    print("------epoch%d------"%(epoch))
    D_true_adv_avg = 0
    D_true_class_avg = 0
    D_fake_adv_avg = 0
    D_fake_class_avg = 0
    D_avg = 0
    G_adv_avg = 0
    G_class_avg = 0
    G_avg = 0
    pro_fake_avg = 0
    pro_true_avg = 0

    dis_count = 0
    for step, (img, label) in enumerate(loader):
        # Generate image
        z = torch.randn(batch_size, inputSize-24, 1, 1)
        noise = torch.cat((z, label.view(batch_size, 24, 1, 1)), 1)

        gen_img = G(noise.cuda(GPUID[0]))
    

        if step%100 == 0 :
            dis_count += 1
            # Update D
            optimizerD.zero_grad()
            pro_true, label_true = D(img.cuda(GPUID[0]))

            pro_fake, label_fake = D(gen_img.cuda(GPUID[0]))
            D_true_adv_loss = criterion(pro_true.view(batch_size, 1), torch.ones(batch_size, 1).cuda(GPUID[0]))
            D_true_class_loss = criterion(label_true, label.cuda(GPUID[0])) * 10
            D_fake_adv_loss = criterion(pro_fake.view(batch_size, 1), torch.zeros(batch_size, 1).cuda(GPUID[0]))

            D_loss = D_true_adv_loss + D_true_class_loss + D_fake_adv_loss
            D_loss.backward()
            optimizerD.step()

            D_avg += D_loss.data.cpu()
            D_true_adv_avg += D_true_adv_loss.data.cpu()
            D_true_class_avg += D_true_class_loss.data.cpu()
            D_fake_adv_avg += D_fake_adv_loss.data.cpu()
        
            pro_fake_avg += pro_fake[0].mean().data.cpu()
            pro_true_avg += pro_true[0].mean().data.cpu()      


        # Update G
        optimizerG.zero_grad()

        pro_true, label_true = D(img.cuda(GPUID[0]))
        D_true_class_loss = criterion(label_true, label.cuda(GPUID[0]))
        gen_img = G(noise.cuda(GPUID[0]))
        pro_fake, label_fake = D(gen_img.cuda(GPUID[0]))
        G_adv_loss = criterion(pro_fake.view(batch_size, 1), torch.ones(batch_size, 1).cuda(GPUID[0]))*0.1
        G_class_loss = criterion(label_fake, label.cuda(GPUID[0])) * 20
        G_loss = G_adv_loss + G_class_loss
        G_loss.backward()
        optimizerG.step()

        G_avg += G_loss.data.cpu()
        G_adv_avg += G_adv_loss.data.cpu()
        G_class_avg += G_class_loss.data.cpu()
        
    G_avg /= step
    G_adv_avg /= step
    G_class_avg /= step


    if dis_count > 0:
        D_avg /= dis_count
        D_true_adv_avg /= dis_count
        D_true_class_avg /= dis_count
        D_fake_adv_avg /= dis_count
        pro_fake_avg /= dis_count
        pro_true_avg /= dis_count
        writer.add_scalar('/Pro/pro_fake_avg', pro_fake_avg, epoch)
        writer.add_scalar('/Pro/pro_true_avg', pro_true_avg, epoch)
        writer.add_scalar('/D/D_avg', D_avg, epoch)
        writer.add_scalar('/D/D_true_adv_avg', D_true_adv_avg, epoch)
        writer.add_scalar('/D/D_true_class_avg', D_true_class_avg, epoch)
        writer.add_scalar('/D/D_fake_adv_avg', D_fake_adv_avg, epoch)

    writer.add_scalar('/G/G_avg', G_avg, epoch)
    writer.add_scalar('/G/G_adv_avg', G_adv_avg, epoch)
    writer.add_scalar('/G/G_class_avg', G_class_avg, epoch)

    iter = epoch + 1
    print('%s (%d %d%%)' % (timeSince(start, iter / epoch_num),
                                         iter, iter / epoch_num * 100))    
    print("G_loss:%f"%(G_avg))
    print("D_loss:%f"%(D_avg))
    print("fake:", pro_fake_avg)
    print("true:", pro_true_avg)

    save_real = torch.zeros(4, 3, 64, 64)
    save_fake = torch.zeros(4, 3, 64, 64)
    for i in range(4):
        save_real[i] =  unnormal(img[i])
        save_fake[i] =  unnormal(gen_img[i])
        
    save_image(save_real, '%sreal_%d.jpg' % (img_save_dir, epoch))
    save_image(save_fake, '%sgen_%d.jpg' % (img_save_dir, epoch))
    torch.save(G.state_dict(), '%sG_%d.pth'%(params_save_dir, epoch))
    torch.save(D.state_dict(), '%sD_%d.pth'%(params_save_dir, epoch))
