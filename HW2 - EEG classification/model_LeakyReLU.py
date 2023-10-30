import torch
import torch.nn as nn
from torch.nn import init


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        firstconv = [   nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
                        nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    ]
        self.firstconv = nn.Sequential(*firstconv)
        depthwiseConv = [   nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16,bias=False),
                            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(),
                            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
                            nn.Dropout(p=0.25)
                        ]
        self.depthwiseConv = nn.Sequential(*depthwiseConv)
        separableConv = [   nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
                            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(),
                            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
                            nn.Dropout(p=0.25)
                        ]
        self.separableConv = nn.Sequential(*separableConv)
        classify = [        nn.Flatten(start_dim=1, end_dim=3),
                            nn.Linear(in_features=736, out_features=2, bias=True)
                        ]
        self.classify = nn.Sequential(*classify)

        self.model = nn.Sequential()
        self.model.add_module("firstconv", self.firstconv)
        self.model.add_module("depthwiseConv", self.depthwiseConv)
        self.model.add_module("separableConv", self.separableConv)
        self.model.add_module("classify", self.classify)

    def forward(self, x):
        return self.model(x)



class DeepCovNet(nn.Module):
    def __init__(self):
        super(DeepCovNet, self).__init__()
        first = [   nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
                    nn.Conv2d(25, 25, kernel_size=(2, 1), bias=True),
                    nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=(1, 2)),
                    nn.Dropout(p=0.5)
                ]
        self.first = nn.Sequential(*first)
        second = [
                    nn.Conv2d(25, 50, kernel_size=(1, 5), bias=True),
                    nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=(1, 2)),
                    nn.Dropout(p=0.5)
                ]
        self.second = nn.Sequential(*second)
        third = [
                    nn.Conv2d(50, 100, kernel_size=(1, 5), bias=True),
                    nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=(1, 2)),
                    nn.Dropout(p=0.5)
                ]
        self.third = nn.Sequential(*third)
        forth = [
                    nn.Conv2d(100, 200, kernel_size=(1, 5), bias=True),
                    nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(kernel_size=(1, 2)),
                    nn.Dropout(p=0.5)
                ]
        self.forth = nn.Sequential(*forth)
        classify = [    nn.Flatten(start_dim=1, end_dim=3),
                        nn.Linear(in_features=8600, out_features=2, bias=True)
                    ]
        self.classify = nn.Sequential(*classify)

        self.model = nn.Sequential()
        self.model.add_module("first", self.first)
        self.model.add_module("second", self.second)
        self.model.add_module("third", self.third)
        self.model.add_module("forth", self.forth)
        self.model.add_module("classify", self.classify)
            

    def forward(self, x):
        return self.model(x)


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



def test(EEG, loader, GPUID, batch_size):
    EEG.eval()
    wrong = 0
    num = 0
    with torch.no_grad():
        for step, (data, label) in enumerate(loader):
            result = EEG(data.float().cuda(GPUID[0]))

            _, indices = torch.max(result, 1)

            wrong += torch.sum(torch.abs(indices.cuda(GPUID[0]) - label.cuda(GPUID[0])))
            num += data.shape[0]
    wrong /= num
    return 1-wrong
