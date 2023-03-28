import torch
import torch.nn as nn
from group import Group




class pdo_net(nn.Module):   
    def __init__(self, n, flip, dis='fd', channel=[16, 24, 32, 32, 48, 64], order=4, g=4, reduction_ratio=1, s=4, drop=0.1, type='regular'):
        super(pdo_net, self).__init__()
        self.flip=flip
        self.channel = channel
        self.type = type
        self.group=Group(n, flip, dis)
        self.conv1=self.group.conv5x5(('trivial', 1), (self.type, self.channel[0]))
        self.bn1=self.group.norm((self.type, self.channel[0]))
        self.bn2=self.group.norm((self.type, self.channel[1]))
        self.bn3=self.group.norm((self.type, self.channel[2]))
        self.bn4=self.group.norm((self.type, self.channel[3]))
        self.bn5=self.group.norm((self.type, self.channel[4]))
        self.bn6=self.group.norm((self.type, self.channel[5]))
        if(g==-1):
            self.conv2=self.group.conv5x5((self.type, self.channel[0]), (self.type, self.channel[1]))
            self.conv3=self.group.conv5x5((self.type, self.channel[1]), (self.type, self.channel[2]))
            self.conv4=self.group.conv5x5((self.type, self.channel[2]), (self.type, self.channel[3]))
            self.conv5=self.group.conv5x5((self.type, self.channel[3]), (self.type, self.channel[4]))
            if(flip):
                self.group=Group(n, False, dis)
                self.restrict=self.group.flip_restrict((self.type, self.channel[4]))
                channel[4]=channel[4]*2
                channel[5]=channel[5]*2
            self.conv6=self.group.conv5x5((self.type,self.channel[4]), (self.type, self.channel[5]))
            


        else:
            self.conv2=self.group.nlpdo((self.type, channel[0]), (self.type, channel[1]), order, reduction=reduction_ratio, s=s, g=g)
            self.conv3=self.group.nlpdo((self.type, channel[1]), (self.type, channel[2]), order, reduction=reduction_ratio, s=s, g=g)
            self.conv4=self.group.nlpdo((self.type, channel[2]), (self.type, channel[3]), order, reduction=reduction_ratio, s=s, g=g)
            self.conv5=self.group.nlpdo((self.type, channel[3]), (self.type, channel[4]), order, reduction=reduction_ratio, s=s, g=g)
            if(flip):
                self.group=Group(n, False, dis)
                self.restrict=self.group.flip_restrict((self.type, channel[4]))
                channel[4]=channel[4]*2
                channel[5]=channel[5]*2
            self.conv6=self.group.nlpdo((self.type, channel[4]), (self.type, channel[5]), order, reduction=reduction_ratio, s=s, g=g)
        
        self.pool=nn.MaxPool2d(2,2)



        self.drop = nn.Dropout2d(drop)


        self.group_pool=self.group.GroupPool((self.type, channel[5]))
        self.global_pool=nn.AdaptiveMaxPool2d(1)

        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(channel[5], 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 10),

        )


    def forward(self, x):
        x=torch.relu(self.bn1(self.conv1(x)))
        x=torch.relu(self.bn2(self.conv2(x)))
        x=self.pool(x)
        x=self.drop(torch.relu(self.bn3(self.conv3(x))))
        x=self.drop(torch.relu(self.bn4(self.conv4(x))))
        x=self.pool(x)
        x=self.drop(torch.relu(self.bn5(self.conv5(x))))
        if(self.flip):
            x=self.restrict(x)

        x=self.drop(torch.relu(self.bn6(self.conv6(x))))
        x=self.group_pool(x)
        x=self.global_pool(x).reshape(x.size(0),-1)

        x=self.fully_net(x)

        return x
        


