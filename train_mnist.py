import torch
import torch.nn as nn
import numpy as np
import time
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from pdo_mnist import pdo_net
from PIL import Image
from datetime import datetime
import argparse





parser = argparse.ArgumentParser(description='net')
parser.add_argument('--model', '-a', default='R', help='Regular(R) or Quotient(Q) ')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training ')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200) ')
parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate (default: 2e-3)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--order', default=4, type=int, help='differential operator order')
parser.add_argument('--bias', default=False, type=bool, help='bias')
parser.add_argument('--reduction', default=1, type=float, help='reduction_ratio')
parser.add_argument('--g', default=4, type=int, help='g * q = z, q:partition number z: number of input fields')
parser.add_argument('--s', default=4, type=int, help='slice, only 1 is valid for quotient model')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout_rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--dis', default='gauss', help='discretization:fd, gauss')
parser.add_argument('--flip', default=False, type=bool, help="D16|5C16 or C16")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = False  # Should make training should go faster for large models
cudnn.deterministic = True
cudnn.enabled = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

print(args)




def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)


totensor = ToTensor()
resize1 = Resize(84)
resize2 = Resize(28)

train_transform = Compose([
    resize1,
    RandomRotation(180),
    resize2,
    totensor
])



mnist_train = MnistRotDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)


test_transform = Compose([
    totensor,
])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)
quotient_type = (('regular', 1),('quo_4', 1))
channel_q=[12, 16, 24, 24, 32, 64]
if args.flip == True:
    channel_r=[12, 16 ,24, 24, 32, 64]
else:
    channel_r=[16, 24 ,32, 32, 48, 64]


if args.model == "R":
    net = pdo_net(16,flip=args.flip ,dis=args.dis, g=args.g,reduction_ratio=args.reduction,drop=args.dropout,s=args.s,order=args.order, channel=channel_r, type='regular')
elif args.model == "Q":
    net = pdo_net(16, False, dis=args.dis,g=args.g, reduction_ratio=args.reduction, drop=args.dropout, s=args.s,
                  order=args.order, type=quotient_type, channel=channel_q)



param=compute_param(net)
model = nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
loss_function = torch.nn.CrossEntropyLoss()



schedule=[60,120,150,180]
learning_rate = args.learning_rate
start=schedule[0]
best=0.
for epoch in range(1,args.epochs):
    if(epoch==schedule[0]):
        learning_rate=learning_rate*0.1
    elif(epoch==schedule[1]):
        learning_rate=learning_rate*0.5
    elif(epoch==schedule[2]):
        learning_rate=learning_rate*0.5
    elif (epoch == schedule[3]):
        learning_rate=learning_rate*0.5


    # t1=time.time()
    print("schedular:{},{},{},{}".format(schedule[0],schedule[1],schedule[2],schedule[3]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    model.train()
    total = 0
    correct = 0
    print('Parameters of the net: {}M'.format(param/(10**6)))
    for i, (x, t) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.cuda()
        t = t.cuda()
        y = model(x)
        _, prediction = torch.max(y.data, 1)
        total += t.shape[0]
        correct += (prediction == t).sum().item()
        loss = loss_function(y, t)
        loss.backward()

        optimizer.step()
    print(f"epoch {epoch} | train accuracy: {correct/total*100.}")
    if(epoch>=start):
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.cuda()
                t = t.cuda()
                
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")
        if(correct/total*100.>best):
            best=correct/total*100

    
    print('\n')
    
print('Best test acc: {}'.format(best))