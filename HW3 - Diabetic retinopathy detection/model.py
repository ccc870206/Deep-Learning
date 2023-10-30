import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        if isinstance(m, nn.Conv2d):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

    return net


class ResNet18(nn.Module):
    """docstring for ResNet18"""
    def __init__(self, is_pretrained):
        super(ResNet18, self).__init__()

        if is_pretrained:
            self.model = models.resnet18(pretrained=True)
            self.model.fc = init_weights(nn.Linear(512, 5))
        else:
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(512, 5)
        
    def forward(self, img):
        return self.model(img)


class ResNet50(nn.Module):
    """docstring for ResNet50"""
    def __init__(self, is_pretrained):
        super(ResNet50, self).__init__()

        if is_pretrained:
            self.model = models.resnet50(pretrained=True)
            self.model.fc = init_weights(nn.Linear(2048, 5))
        else:
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(2048, 5)
        
    def forward(self, img):
        return self.model(img)


def test(ResNet, loader, GPUID, batch_size):
    ResNet.eval()
    acc = 0
    num = 0

    with torch.no_grad():
        for step, (data, label) in enumerate(loader):
            result = ResNet(data.float().cuda(GPUID[0]))

            _, indices = torch.max(result, 1)

            acc += torch.sum(torch.eq(indices.cuda(GPUID[0]), label.cuda(GPUID[0])).int()).cpu().numpy()
            num += data.shape[0]
            # break
    acc /= num
    return acc