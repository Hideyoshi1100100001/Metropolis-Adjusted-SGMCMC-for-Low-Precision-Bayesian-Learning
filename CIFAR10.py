import math
import time
import os
import shutil

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import logging
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torchvision import datasets, transforms
import scipy.stats

cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
print(cuda_device)

parser = argparse.ArgumentParser(description='ResNets with Metropolis-Hastings for CIFAR-10')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end-epoch', default=0, type=int, metavar='N',
                    help='manual end epoch number')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--temperature', default=1e-8, type=float, help='factor that multiplies gaussian noise, momentum variance')
parser.add_argument('--annealing', default=1, type=float, help='(0, 1], the lower bound of anneal temperature, default 1(disable)')
parser.add_argument('--momentumDecay', default=0.1, type=float, metavar='M',
                    help='momentum decay')
parser.add_argument('--priorSigma', '--pS', default=1e2, type=float,
                    metavar='P', help='equivalent to weight decay (default: 1e2 -> 1e-4)')
parser.add_argument('--quantizeDecay', '--qd', default=0, type=float, help='decay the real weight to the quantized (default: 0, disable)')
parser.add_argument('--LoadPath', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--SavePath', help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--saveInterval',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)
parser.add_argument('--sampleNumber', default=10, type=int, help='sample number for ensemble')
parser.add_argument('--sampleInterval', default=1, type=int, help='epochs between two samples')
parser.add_argument('--deepEnsemble', default=1, type=int, help='rerun the training')
parser.add_argument('--MH', action='store_true', help='MH or pure SGLD')
parser.add_argument('--MHInterval', default=5, type=int, help='iterations between two MH tests')
parser.add_argument('--MHEpsilon', default=2e-1, type=float, help='[0,1], larger->easier to pass MH test')
parser.add_argument('--transitionKernel', '--tk', type=float, default=[1, 1], nargs='+', help='possibility of using different transition kernel, [0]: default, [1] - [0]: flip')
parser.add_argument('--flipPossibility', default=0.05, type=float, help='[0,1], possibility of flipping element wise when transision kernel is random flip')
parser.add_argument('--fixed_lr', action='store_true', help='fix the learning rate')
parser.add_argument('--debug_show_prob', action='store_true', help='show the probability about MH test')
parser.add_argument('--ecebins', default=15, type=int, help='numbers of ECE bins')

args = parser.parse_args()

if args.end_epoch == 0:
    args.end_epoch = args.epochs

if not os.path.exists(args.SavePath):
    os.makedirs(args.SavePath)

if args.LoadPath != '':
    if (args.LoadPath != args.SavePath):
        shutil.copyfile(os.path.join(args.LoadPath, "log.txt"), os.path.join(args.SavePath, "log.txt"))
    log = open(os.path.join(args.SavePath, "log.txt"),"a+")
else:
    log = open(os.path.join(args.SavePath, "log.txt"),"w")
    log.flush()

print("args:\n")
for (k, v) in vars(args).items():
    line = "{}:\t{}".format(k, v)
    print(line)
    log.write(line+"\n")
    log.flush()

batchsize = args.batch_size  # training batch size
infeature, outfeature = 32*32, 10

def mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, num_workers=1
    )

    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=1
    )
    return trainloader, testloader

def cifar():
    rgb_mean = np.array([0.4914, 0.4822, 0.4465])
    rgb_std = np.array([0.2023, 0.1994, 0.2010])
    if 1:  #data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, num_workers=1)
    datasize = trainset.data.shape[0]

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=500, shuffle=False, num_workers=1)
    return datasize, trainloader, testloader

datasize, trainloader, testloader = cifar()
print("datasize: ",datasize)

def Log_UP(epoch, total_epochs):
    K_min, K_max = 1e-3, 1e1 
    # K_min, K_max = 1e-1, 1e1
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / total_epochs * epoch)]).float().cuda()


class BinaryQuantize(torch.autograd.Function):
    '''
        binary quantize function, from IR-Net
        (https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w1a/modules/binaryfunction.py)
    ''' 

    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k, t = k.cuda(), t.cuda() 
        grad_input = k * t * (1-torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class Maxout(nn.Module):
    '''
        Nonlinear function
    '''

    def __init__(self, channel, neg_init=0.25, pos_init=1.0):
        super(Maxout, self).__init__()
        self.neg_scale = nn.Parameter(neg_init*torch.ones(channel))
        self.pos_scale = nn.Parameter(pos_init*torch.ones(channel))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Maxout
        x = self.pos_scale.view(1,-1,1,1)*self.relu(x) - self.neg_scale.view(1,-1,1,1)*self.relu(-x)
        return x

class BinaryActivation(nn.Module):
    '''
        learnable distance and center for activation
    '''

    def __init__(self):
        super(BinaryActivation, self).__init__() 
        self.alpha_a = nn.Parameter(torch.tensor(1.0))
        self.beta_a = nn.Parameter(torch.tensor(0.0))

    def gradient_approx(self, x):
        '''
            gradient approximation
            (https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal18_34/birealnet.py)
        '''
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

    def forward(self, x):
        x = (x-self.beta_a)/self.alpha_a
        x = self.gradient_approx(x)
        return self.alpha_a*(x + self.beta_a)


class LambdaLayer(nn.Module):
    '''
        for DownSample
    '''

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class AdaBin_Conv2d(nn.Conv2d):
    '''
        AdaBin Convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False, a_bit=1, w_bit=1):
        super(AdaBin_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.k = torch.tensor([10]).float().cpu()
        self.t = torch.tensor([0.1]).float().cpu() 
        self.binary_a = BinaryActivation()

        self.filter_size = self.kernel_size[0]*self.kernel_size[1]*self.in_channels

    def forward(self, inputs):
        if self.a_bit == 1:
            inputs = self.binary_a(inputs) 

        if self.w_bit == 1:
            w = self.weight 
            beta_w = w.mean((1, 2, 3)).view(-1, 1, 1, 1)
            alpha_w = torch.sqrt(((w-beta_w)**2).sum((1, 2, 3)) / self.filter_size).view(-1, 1, 1, 1)

            w = (w - beta_w)/alpha_w
            wb = BinaryQuantize().apply(w, self.k, self.t)
            weight = wb * alpha_w + beta_w
        else:
            weight = self.weight

        output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = AdaBin_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = Maxout(planes)

        self.conv2 = AdaBin_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear2 = Maxout(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                     )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = self.nonlinear1(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = self.nonlinear2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sample=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = Maxout(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(512*block.expansion)

        self.predWeight = None
        self.sampling = sample

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        # out = F.softmax(out, dim=1)
        if(self.sampling):
            return F.softmax(out, dim=1) * self.predWeight
        return out

def net(sample=False):
    return ResNet(BasicBlock_1w1a, [2,2,2,2], sample=sample)


class SGHLDLP_F(Optimizer):
    def __init__(
        self,
        optim,
        grad_scaling=1.0,
        noise=False,
        temperature=1.0,
        datasize=None,
        mDecay=args.momentumDecay
    ):
        assert isinstance(optim, torch.optim.SGD)
        super(SGHLDLP_F, self).__init__(optim.param_groups, optim.defaults)
        self.param_groups = optim.param_groups
        self.optim = optim

        assert grad_scaling > 0
        self.grad_scaling = grad_scaling

        self.k = None
        self.t = None
        self.annealing = 1.
        self.mGroups = []
        mGroupsIndex = -1
        for group in self.param_groups:
            self.mGroups.append([])
            mGroupsIndex += 1
            for p in group["params"]:
                #p.data = stochastic_quantizer(p)
                p.grad = torch.zeros(p.shape).to(cuda_device)
                self.mGroups[mGroupsIndex].append(torch.randn(p.shape).to(cuda_device) * np.sqrt(args.lr / datasize))
        self.noise = noise
        self.temperature = temperature
        self.datasize = datasize
        self.mDecay = mDecay

        self.predProb = None
        self.dataProb = 0.1
        self.paramProb = None
        self.lastM = 1
        self.lastP = [[None for _ in group["params"]] for group in self.param_groups]

    def getTemperature(self, temp):
        self.temperature = temp

    def step(self, lr=None, half=False):
        #print("==================================")
        if self.paramProb is None:
            self.paramProb = 0
        for i, (group, mGroup) in enumerate(zip(self.param_groups, self.mGroups)):
            if lr:
                group["lr"] = lr
            dist = 0
            for j, (p, mp) in enumerate(zip(group["params"], mGroup)):
                d_p = p.grad.data + p / (args.priorSigma ** 2)

                if(args.quantizeDecay > 0 and len(p.shape) == 4):
                    filter_size = p.shape[1] * p.shape[2] * p.shape[3]
                    beta_w = p.mean((1, 2, 3)).view(-1, 1, 1, 1)
                    alpha_w = torch.sqrt(((p-beta_w)**2).sum((1, 2, 3)) / filter_size).view(-1, 1, 1, 1)
                    w = (p - beta_w)/alpha_w
                    wb = BinaryQuantize().apply(w, self.k, self.t)
                    weight = wb * alpha_w + beta_w
                    d_p.add_(weight - p, alpha=args.quantizeDecay)

                temp2 = p
                mp.data.add_(mp, alpha=-self.mDecay)
                mp.data.add_(d_p, alpha=(- group["lr"]) / self.annealing)
                temp = mp / (1 + self.mDecay)

                randomValue = torch.rand(1).item()
                if (randomValue <= args.transitionKernel[0] or len(p.shape) != 4): #SGHMC, gradient based method
                    if (self.noise):
                        eps = torch.randn(p.size()).to(cuda_device)
                        noise = (
                            group["lr"] * args.temperature * self.mDecay
                        ) ** 0.5 * 2 * eps
                        #print(torch.norm(newp).item()/torch.norm(noise).item())
                        mp.data.add_(noise)
                    mp /= 1 + self.mDecay
                    p.data.add_(mp, alpha=(0.5 if half else 1) * group["lr"])
                elif (randomValue <= args.transitionKernel[1]): #random flip, non-gradient method
                    beta_w = p.mean((1, 2, 3)).view(-1, 1, 1, 1)
                    mask = (torch.rand(beta_w.shape) <= args.flipPossibility).float().to(cuda_device)
                    p.data.add_((beta_w - p) * mask, alpha=2)

                if(args.MH):
                    if (randomValue <= args.transitionKernel[0]):
                        dist1 = torch.norm(temp2 + temp * (0.5 if half else 1) - p)**2 #q(θ_t+1|θ_t)
                        if(self.lastP[i][j] is not None):
                            dist2 = torch.norm(temp2 + temp * self.lastM - self.lastP[i][j])**2 #q(θ_t|θ_t+1)
                        else:
                            dist2 = torch.zeros(1)
                        self.lastP[i][j] = temp2
                        self.lastM = 0.5 if half else 1
                        dist += (dist2.item() - dist1.item()) #/ p.numel()
                        #dist += torch.sum(d_p * (temp + mp)).item() / 2
                    elif (randomValue <= args.transitionKernel[1]): #non-gradient method
                        dist1 = torch.norm(temp2 + temp * (0.5 if half else 1) - p)**2 #q(θ_t+1|θ_t)
                        if(self.lastP[i][j] is not None):
                            dist2 = torch.norm(temp2 + temp * self.lastM - self.lastP[i][j]) ** 2
                        else:
                            dist2 = torch.zeros(1)
                        self.lastP[i][j] = temp2
                        self.lastM = 0.5 if half else 1
                        dist += (dist2.item() - dist1.item())
            if(args.MH and (args.temperature > 0)):
                #self.paramProb -= dist
                self.paramProb += dist / (4 * group["lr"] * args.temperature * args.momentumDecay)
        #print("==================================")
        #print("\nend\n")

    def getParamProb(self):
        for i, (group, mGroup) in enumerate(zip(self.param_groups, self.mGroups)):
            dist = 0
            for j, (p, mp) in enumerate(zip(group["params"], mGroup)):
                temp = p.grad.data + p / (args.priorSigma ** 2)

                if(len(p.shape) == 4):
                    filter_size = p.shape[1] * p.shape[2] * p.shape[3]
                    beta_w = p.mean((1, 2, 3)).view(-1, 1, 1, 1)
                    alpha_w = torch.sqrt(((p-beta_w)**2).sum((1, 2, 3)) / filter_size).view(-1, 1, 1, 1)
                    w = (p - beta_w)/alpha_w
                    wb = BinaryQuantize().apply(w, self.k, self.t)
                    weight = wb * alpha_w + beta_w
                    temp.add_(weight - p, alpha=args.quantizeDecay)

                temp = ((1 - self.mDecay) * mp - temp * group["lr"]) / (1 + self.mDecay)
                if(self.lastP[i][j] is not None):
                    dist2 = torch.norm(p + temp * self.lastM - self.lastP[i][j])**2 #q(θ_t|θ_t+1)
                else:
                    dist2 = torch.zeros(1)
                self.lastP[i][j] = None
                self.lastM = 1
                dist += (dist2.item()) #/ p.numel()
            if(args.MH and (args.temperature > 0)):
                #self.paramProb -= dist
                self.paramProb += dist / (4 * group["lr"] * args.temperature * args.momentumDecay)
        temp = self.paramProb
        self.paramProb = 0
        return temp

    def getPrior(self):
        temp = 0
        for _, group in enumerate(self.param_groups):
            for _, p in enumerate(group["params"]):
                temp += (torch.norm(p).item() ** 2)
        return temp


def Metropolis_Hastings(Ocur, Onext, Mcur, Mnext):
    accept = False
    Onext.dataProb = 0
    paramProb = 5000000
    if Ocur.paramProb is None:
        Ocur.paramProb = 5000000
    else:
        paramProb = Onext.getParamProb()
    total = 0
    randomValue = torch.rand(1).item()

    ell = None
    priorTerm = (Onext.getPrior() - Ocur.getPrior()) / (2 * args.priorSigma ** 2) / Onext.annealing
    mu_0 = (math.log(randomValue) + paramProb + priorTerm) / datasize
    if(args.debug_show_prob):
        #print("log_paramProb:", paramProb / datasize)
        #print("mu_0:", mu_0)
        print("q term:\t", math.exp(-paramProb))
        print("prior term:\t", math.exp(-priorTerm))
    with torch.no_grad():
        for (data, target) in trainloader:

            data, target = Variable(data), Variable(target)
            data = data.to(cuda_device)
            target = target.to(cuda_device)
            total += target.size(0)

            output = F.softmax(Mnext(data), dim=1)
            Onext.predProb = torch.log(output.data[torch.arange(target.size(0)),target]) / Onext.annealing
            output = F.softmax(Mcur(data), dim=1)
            Ocur.predProb = torch.log(output.data[torch.arange(target.size(0)),target]) / Onext.annealing

            if ell is None:
                ell = Onext.predProb - Ocur.predProb
            else:
                ell = torch.cat((ell, Onext.predProb - Ocur.predProb), 0)
            s_l = (ell.std().item() * math.sqrt(total / (total - 1)))
            s = s_l * math.sqrt((datasize - total) / (total * (datasize - 1)))
            Onext.dataProb += (Onext.predProb - Ocur.predProb).sum()
            if(args.debug_show_prob):
                pass
            if(s != 0):
                delta = 1 - scipy.stats.t.cdf(abs(ell.mean().item() - mu_0) / s, total - 1)
                if (delta < args.MHEpsilon):
                    if(args.debug_show_prob):
                        #print(delta)
                        print("p term:\t", torch.exp(datasize * ell.mean()).item())
                        print("accept rate:\t", math.exp(-paramProb - priorTerm) * torch.exp(datasize * ell.mean()).item())
                    return (ell.mean().item() > mu_0)
    Onext.dataProb = Onext.dataProb.item() / total
    if(Onext.dataProb > mu_0):
        accept = True
    return accept


def train(start_epoch, best_prec, model, train_loader, test_loader, optimizer, lr_scheduler, args, cuda_device):
    i = 0
    val_accuracy = 0
    criterion = nn.CrossEntropyLoss().cuda()

    train_acc_list = []
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    train_ece_list = []
    test_ece_list = []
    eceBin = [x * 1.0001 / args.ecebins for x in range(args.ecebins + 1)]
    sampleList = [net(sample=True).to(cuda_device) for _ in range(args.sampleNumber)]
    sampleId = 0

    bestModel = net().to(cuda_device)
    bestOpt = torch.optim.SGD(bestModel.parameters(), lr=args.lr)
    bestOpt = SGHLDLP_F(
        bestOpt,
        noise=True,
        temperature=args.temperature,
        datasize=50000,
        mDecay=args.momentumDecay
    )
    if(args.LoadPath != ''):
        checkpoint = torch.load(os.path.join(args.LoadPath, "checkpoint.pth"))
        bestModel.load_state_dict(checkpoint['state_dict'])
        bestOpt.load_state_dict(checkpoint['optimizer'])
    if(args.fixed_lr):
        for group in bestOpt.param_groups:
            group["lr"] = args.lr if group["lr"] is not None else None
        Blr_scheduler = torch.optim.lr_scheduler.StepLR(bestOpt, step_size=1e8, gamma=1, last_epoch=-1)
    else:
        Blr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(bestOpt, args.epochs, eta_min = 0, last_epoch=-1)
    for epoch in range(start_epoch):
        Blr_scheduler.step()
    for epoch in range(start_epoch, args.end_epoch):
        MH_total, MH_accept = 0., 0.
        tic = time.time()
        model.train()
        train_loss = 0
        total = 0
        correct = 0
        eceTotal = [0 for _ in range(args.ecebins)]
        ecePred = [0 for _ in range(args.ecebins)]
        eceAcc = [0 for _ in range(args.ecebins)]
        ece = 0
        line = "\nEpoch: {}\tLearningRate: {}\tStepTemperature:{}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.temperature)
        print(line)
        log.write(line + "\n")
        log.flush()

        t = Log_UP(epoch, args.epochs)
        annealTemp = (1 + args.annealing) / 2 + (1 - args.annealing) / 2 * math.cos(epoch * math.pi / args.epochs)
        optimizer.annealing = annealTemp
        bestOpt.annealing = annealTemp
        if (t < 1):
            k = 1 / t
        else:
            k = torch.tensor([1]).float().cuda()
        for m in model.modules():
            if isinstance(m, AdaBin_Conv2d):
                m.t = t
                m.k = k
        optimizer.t = t
        optimizer.k = k
        bestOpt.t = t
        bestOpt.k = k

        with tqdm.tqdm(enumerate(trainloader), total=len(trainloader)) as t:
            t.set_description(f"Epoch {epoch} train")
            for batch_idx, (data, target) in t:
                i += 1

                data, target = Variable(data), Variable(target)
                data = data.cuda()
                target = target.cuda()

                output = model(data)
                loss = criterion(output, target)
                prediction = output.data.max(1)[1]
                optimizer.zero_grad()
                loss.backward()

                if (False == args.MH):
                    optimizer.step()
                else:
                    if (0 != (batch_idx+1) % args.MHInterval):
                        #optimizer.step(half=(True if (1 == (batch_idx+1) % args.MHInterval) else False))
                        optimizer.step(half=False)
                    else:
                        accept = Metropolis_Hastings(bestOpt, optimizer, bestModel, model)
                        MH_total += 1
                        if accept:
                            MH_accept += 1
                            torch.save({'state_dict':model.state_dict(),
                                       'optimizer':optimizer.state_dict()}, os.path.join(args.SavePath, "temp.pth"))
                            checkpoint = torch.load(os.path.join(args.SavePath, "temp.pth"))
                            bestOpt.load_state_dict(checkpoint['optimizer'])
                            bestModel.load_state_dict(checkpoint['state_dict'])
                        else:
                            bestOpt.mGroups = [[-i for i in group] for group in bestOpt.mGroups]
                            torch.save({'state_dict':bestModel.state_dict(),
                                       'optimizer':bestOpt.state_dict()}, os.path.join(args.SavePath, "temp.pth"))
                            checkpoint = torch.load(os.path.join(args.SavePath, "temp.pth"))
                            optimizer.load_state_dict(checkpoint['optimizer'])
                            model.load_state_dict(checkpoint['state_dict'])
                        optimizer.step(half=False)

                train_loss += loss.item() * target.size(0)
                total += target.size(0)
                correct += prediction.eq(target.data).sum().item()
                
                #calculate ECE

                predVal = torch.gather(F.softmax(output, dim=1), 1, target.view(-1, 1))
                ece = 0
                for ecei in range(args.ecebins):
                    predPos = (predVal >= eceBin[ecei]) * (predVal < eceBin[ecei + 1])
                    eceTotal[ecei] += predPos.sum().item()
                    ecePred[ecei] += (predVal * predPos).sum().item()
                    eceAcc[ecei] += (prediction.eq(target.data).view(-1, 1) * predPos).sum().item()
                    ece += abs(ecePred[ecei] - eceAcc[ecei])

                t.set_postfix(
                    {
                        "loss": f"{train_loss/total:.3f}",
                        "acc": f"{100.*correct/total:.3f}%, {correct}/{total}",
                        "ECE": f"{ece/total:.3f}",
                        "accept": f"{(MH_accept/MH_total if MH_total > 0 else 0):.3f}"
                    }
                )

        lr_scheduler.step()
        Blr_scheduler.step()
        t = time.time() - tic

        if (args.sampleInterval - 1 == epoch % args.sampleInterval):
            torch.save(model.state_dict(), os.path.join(args.SavePath, "temp.pth"))
            sampleList[sampleId].load_state_dict(torch.load(os.path.join(args.SavePath, "temp.pth")))
            sampleList[sampleId].sampling = True
            sampleList[sampleId].predWeight = optimizer.param_groups[0]["lr"]
            sampleId = (sampleId + 1) % args.sampleNumber

        # validate
        if 0:
            for p in model.parameters():
                print(p[0])
                break
        val_accuracy, val_loss, val_ece = evaluate(model, test_loader, epoch)
        if (val_accuracy > best_prec):
            best_prec = val_accuracy

        if (False == args.MH):
            line = "Epoch: {}\tTime: {:.1f}s\tTrain Loss: {:.3f}\tTrain Acc: {:.3f}%\tTrain ECE: {:.3f}\nVal Loss: {:.3f}\tVal Acc: {:.3f}\tVal ECE: {:.3f}\tBest Acc: {:.3f}".format(
                epoch, t,train_loss/total, 100*correct/total, ece/total, val_loss, val_accuracy, val_ece, best_prec)
            print(line)
            log.write(line + "\n")
            log.flush()
        else:
            line = "Epoch: {}\tTime: {:.1f}s\tTrain Loss: {:.3f}\t Train Acc: {:.3f}%\tTrain ECE: {:.3f}\nVal Loss: {:.3f}\tVal Acc: {:.3f}\tVal ECE: {:.3f}\tBest Acc: {:.3f}\tMH Accept Rate: {:.3f}".format(
                epoch, t,train_loss/total, 100*correct/total, ece/total, val_loss, val_accuracy, val_ece, best_prec, MH_accept/MH_total)
            print(line)
            log.write(line + "\n")
            log.flush()
        if(args.SavePath != '' and ((epoch + 1) % args.saveInterval == 0)):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec': best_prec,
            }, os.path.join(args.SavePath, "checkpoint.pth"))
            print("ckpt saved complete to " + os.path.join(args.SavePath, "checkpoint.pth"))
        train_loss_list.append(train_loss/total)
        train_acc_list.append(100*correct/total)
        test_loss_list.append(val_loss)
        test_acc_list.append(val_accuracy)
        train_ece_list.append(ece/total)
        test_ece_list.append(val_ece)

    print("Total number of steps: {}".format(i))
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, sampleList


def evaluate(model, test_loader, epoch):
    model.eval()
    val_acc = 0
    val_loss = 0
    total = 0
    criterion = nn.CrossEntropyLoss().cuda()

    eceBin = [x * 1.0001 / args.ecebins for x in range(args.ecebins + 1)]
    eceTotal = [0 for _ in range(args.ecebins)]
    ecePred = [0 for _ in range(args.ecebins)]
    eceAcc = [0 for _ in range(args.ecebins)]
    ece = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            val_loss += criterion(output, target) * target.size(0)
            total += target.size(0)
            prediction = output.data.max(1)[1]

            #calculate ECE

            predVal = torch.gather(F.softmax(output, dim=1), 1, target.view(-1, 1))
            for ecei in range(args.ecebins):
                predPos = (predVal >= eceBin[ecei]) * (predVal < eceBin[ecei + 1])
                eceTotal[ecei] += predPos.sum().item()
                ecePred[ecei] += (predVal * predPos).sum().item()
                eceAcc[ecei] += (prediction.eq(target.data).view(-1, 1) * predPos).sum().item()
                ece += abs(ecePred[ecei] - eceAcc[ecei])

            val_acc += prediction.eq(target.data).sum().item()
    
    if epoch == args.end_epoch - 1:
        for ecei in range(args.ecebins):
            if eceTotal[ecei] > 0:
                line = "ECE bin [{:.3f}, {:.3f}], pred / acc / total: {:.3f} / {:.3f} / {}".format(eceBin[ecei], eceBin[ecei + 1], ecePred[ecei] / eceTotal[ecei],
                                                                                            eceAcc[ecei] / eceTotal[ecei], eceTotal[ecei])
            else:
                line = "ECE bin [{:.3f}, {:.3f}], pred / acc / total: 0 / 0 / 0".format(eceBin[ecei], eceBin[ecei + 1])
            print(line)
            log.write(line + "\n")
            log.flush()

    return 100. * val_acc / total, val_loss / total, ece / total

samples = []

start_time = time.time()

for _ in range(args.deepEnsemble):
    model = net().to(cuda_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = SGHLDLP_F(
        optimizer,
        noise=True,
        temperature=args.temperature,
        datasize=50000,
        mDecay=args.momentumDecay
    )
    start_epoch = args.start_epoch
    best_prec = 0
    if(args.LoadPath != ''):
        line = "Loading checkpoint '{}'".format(os.path.join(args.LoadPath, "checkpoint.pth"))
        log.write(line + "\n")
        log.flush()
        print(line)
        checkpoint = torch.load(os.path.join(args.LoadPath, "checkpoint.pth"))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            best_prec = checkpoint['best_prec']
            start_epoch = checkpoint['epoch']
            args.start_epoch = start_epoch
        except:
            pass
        line = "Loaded checkpoint '{}' (epoch {})".format(os.path.join(args.LoadPath, "checkpoint.pth"), start_epoch)
        print(line)
        log.write(line + "\n")
        log.flush()
        if(0):
            model.initFromFullPresicion()
        for group in optimizer.param_groups:
            group["lr"] = args.lr if group["lr"] is not None else None
    if(args.fixed_lr):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1e8, gamma=1, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)
    for epoch in range(args.start_epoch):
        lr_scheduler.step()
    train_loss, train_acc, test_loss, test_acc, sample = train(start_epoch, best_prec, model, trainloader, testloader, optimizer, lr_scheduler, args, cuda_device)
    samples += sample

eceBin = [i * 1.0001 / args.ecebins for i in range(args.ecebins + 1)]
eceTotal = [0 for _ in range(args.ecebins)]
ecePred = [0 for _ in range(args.ecebins)]
eceAcc = [0 for _ in range(args.ecebins)]
ece = 0

#_, _, _ = evaluate(samples[0], testloader)
with tqdm.tqdm(enumerate(testloader), total=len(testloader)) as t:
    t.set_description(f"Sample test")
    fac = 0
    for sample in samples:
        sample.eval()
        fac += sample.predWeight
    correct = 0
    total = 0
    loss = 0
    criterion = nn.NLLLoss().cuda()
    with torch.no_grad():
        for batch_idx, (data, target) in t:
            output = None
            data, target = Variable(data), Variable(target)
            data = data.cuda()
            target = target.cuda()
            for sample in samples:
                if output is None:
                    output = sample(data)
                else:
                    output += sample(data)

            output /= fac
            loss += criterion(torch.log(output), target) * target.size(0)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum().item()
            total += target.size(0)

            #calculate ECE

            predVal = torch.gather(output, 1, target.view(-1, 1))
            for ecei in range(0, args.ecebins):
                predPos = (predVal >= eceBin[ecei]) * (predVal < eceBin[ecei + 1])
                eceTotal[ecei] += predPos.sum().item()
                ecePred[ecei] += (predVal * predPos).sum().item()
                eceAcc[ecei] += (prediction.eq(target.data).view(-1, 1) * predPos).sum().item()
                ece += abs(ecePred[ecei] - eceAcc[ecei])

            t.set_postfix(
                {
                    "loss": f"{loss/total:.3f}",
                    "acc": f"{100.*correct/total:.3f}%, {correct}/{total}",
                    "ece": f"{ece/total:.3f}",
                }
            )
    line = "Sample test: loss: {:.3f}, acc: {:.3f}%, ece: {:.3f}".format(loss.item()/total, 100*correct/total, ece/total)
    print(line)
    log.write(line + "\n")
    log.flush()
    for ecei in range(args.ecebins):
        if eceTotal[ecei] > 0:
            line = "ECE bin [{:.3f}, {:.3f}], pred / acc / total: {:.3f} / {:.3f} / {}".format(eceBin[ecei], eceBin[ecei + 1], ecePred[ecei] / eceTotal[ecei],
                                                                                        eceAcc[ecei] / eceTotal[ecei], eceTotal[ecei])
        else:
            line = "ECE bin [{:.3f}, {:.3f}], pred / acc / total: 0 / 0 / 0".format(eceBin[ecei], eceBin[ecei + 1])
        print(line)
        log.write(line + "\n")
        log.flush()

line = "Total time: {}s".format(time.time() - start_time)
print(line)
log.write(line + "\n\n")
log.write("==================================================\n")
log.flush()