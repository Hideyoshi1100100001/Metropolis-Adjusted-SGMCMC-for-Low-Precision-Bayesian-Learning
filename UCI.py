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
from numpy.random import RandomState
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from torch.hub import load_state_dict_from_url
from torchvision import datasets, transforms
from collections import OrderedDict
import scipy.stats

cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
print(cuda_device)

parser = argparse.ArgumentParser(description='MLPs with Metropolis-Hastings for UCI')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end-epoch', default=0, type=int, metavar='N',
                    help='manual end epoch number')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--temperature', default=1e-8, type=float, help='factor that multiplies gaussian noise, momentum variance')
parser.add_argument('--annealing', default=1, type=float, help='(0, 1], the lower bound of anneal temperature, default: 1(disable)')
parser.add_argument('--momentumDecay', default=0.1, type=float, metavar='M',
                    help='momentum decay, default: 0.1')
parser.add_argument('--priorSigma', '--pS', default=316.2, type=float,
                    metavar='P', help='equivalent to weight decay (default: 316.2 -> 1e-5)')
parser.add_argument('--LoadPath', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--SavePath', help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--saveInterval',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)
parser.add_argument('--sampleNumber', default=10, type=int, help='sample number for ensemble')
parser.add_argument('--sampleInterval', default=1, type=int, help='epochs between two samples')
parser.add_argument('--rerun', default=1, type=int, help='rerun the training')
parser.add_argument('--MH', action='store_true', help='MH or pure SGHMC')
parser.add_argument('--MHInterval', default=5, type=int, help='iterations between two MH tests')
parser.add_argument('--MHEpsilon', default=2e-1, type=float, help='[0,1], larger->easier to pass MH test')
parser.add_argument('--fixed_lr', action='store_true', help='fix the learning rate')
parser.add_argument('--debug_show_prob', action='store_true', help='show the probability about MH test')
parser.add_argument('--dataPath', help='The directory of UCI dataset',
                    default='', type=str)

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

outDict = {'adult': 2, 'abalone': 1, 'wine': 3, 'yacht': 1, 'Concrete': 1, 'DryBean': 7, 'HousingData': 1}

outfeature = outDict[args.dataPath]
def UCI():
    if(args.dataPath in ['adult', 'abalone', 'wine', 'yacht']):
        path = "./data/" + args.dataPath + ".data"
        rawData = pd.read_csv(path, sep=',')
        rawData = pd.read_csv(path, sep=',', names=[i for i in range(rawData.shape[1])])
    else:
        path = "./data/" + args.dataPath + ".csv"
        rawData = pd.read_csv(path, sep=',')
        rawData = pd.read_csv(path, sep=',', names=[i for i in range(rawData.shape[1])])
    rawData = rawData.sample(frac=1)

    if(args.dataPath in ['adult', 'abalone', 'yacht', 'Concrete', 'DryBean']):
        features = rawData.iloc[:, :-1]
        labels = rawData.iloc[:, -1]
    else:
        features = rawData.iloc[:, 1:]
        labels = rawData.iloc[:, 0]

    if(outfeature == 1):
        labels = torch.tensor(labels.values.astype(float), dtype=torch.float)
    else:
        labels = torch.tensor(labels.values.astype(float), dtype=torch.int64)
    if(args.dataPath in ['wine',]):
        labels = labels - 1

    numeric_features = features.dtypes[features.dtypes != 'object'].index

    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    features[numeric_features] = features[numeric_features].fillna(0)
    features = pd.get_dummies(features, dummy_na=False)
    features = torch.tensor(features.values.astype(float), dtype=torch.float)

    if 0:
        print(features.shape, features[0])
        print(labels.shape, labels[:10])

    all, n = features.shape[0], int(features.shape[0] / 6)

    #print(all - n)

    testset = torch.utils.data.TensorDataset(features[:n], labels[:n])
    testloader = torch.utils.data.DataLoader(testset, 256, shuffle=False)
    trainset = torch.utils.data.TensorDataset(features[n:], labels[n:])
    trainloader = torch.utils.data.DataLoader(trainset, batchsize, shuffle=True)
    datasize = all - n
    alldataset = torch.utils.data.TensorDataset(features, labels)
    alldataloader = torch.utils.data.DataLoader(alldataset, all, shuffle=False)

    return datasize, trainloader, testloader, alldataloader, features.shape[1]

datasize, trainloader, testloader, _, infeature = UCI()
print("datasize: ",datasize)

def Log_UP(epoch, total_epochs):
    K_min, K_max = 1e-3, 1e1 
    # K_min, K_max = 1e-1, 1e1
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / total_epochs * epoch)]).float().cuda()


class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=False):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2)  #/ (self.thd_pos ** 0.5))
        else:
            self.s = nn.Parameter(x.detach().abs().mean() * 2) #/ (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / x.numel() ** 0.5  #((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / x.numel() ** 0.5  #((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


class QuanLinear(nn.Linear):
    def __init__(self, m: nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return nn.functional.linear(quantized_act, quantized_weight, self.bias)

class MLP(nn.Module):
    def __init__(self, hidden=1024, sample=False):

        super(MLP, self).__init__()

        self.in_features = infeature
        self.hidden = hidden
        self.out_features = outfeature
        self.quant = LsqQuan(2)

        self.linear1 = nn.Linear(self.in_features, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.out_features)
        self.quant1 = LsqQuan(2)
        self.quant1.init_from(self.linear1.weight)
        self.quant2 = LsqQuan(2)
        self.quant2.init_from(self.linear2.weight)

        self.predWeight = None
        self.sampling = sample

    def forward(self, x):
        #print(self.linear1.weight)
        quantized_weight1 = self.quant1(self.linear1.weight)
        quantized_weight2 = self.quant2(self.linear2.weight)
        x = F.linear(x.view(-1, self.in_features), quantized_weight1, self.linear1.bias)
        x = F.relu(x)
        if self.quant is not None:
            x = self.quant(x)
        x = F.linear(x, quantized_weight2, self.linear2.bias)
        if(self.out_features > 1):
            if(self.sampling):
                return x * self.predWeight
            return x
        else:
            if(self.sampling):
                return x.view(-1) * self.predWeight
            return x.view(-1)


class SGHMC(Optimizer):
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
        super(SGHMC, self).__init__(optim.param_groups, optim.defaults)
        self.param_groups = optim.param_groups
        self.optim = optim

        assert grad_scaling > 0
        self.grad_scaling = grad_scaling

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
        if self.paramProb is None:
            self.paramProb = 0
        for i, (group, mGroup) in enumerate(zip(self.param_groups, self.mGroups)):
            if lr:
                group["lr"] = lr
            dist = 0
            for j, (p, mp) in enumerate(zip(group["params"], mGroup)):
                d_p = p.grad.data + p / (args.priorSigma ** 2)

                temp2 = p
                mp.data.add_(mp, alpha=-self.mDecay)
                mp.data.add_(d_p, alpha=(- group["lr"]) / self.annealing)
                temp = mp / (1 + self.mDecay)

                if (self.noise):
                    eps = torch.randn(p.size()).to(cuda_device)
                    noise = (
                        group["lr"] * args.temperature * self.mDecay
                    ) ** 0.5 * 2 * eps
                    mp.data.add_(noise)
                mp /= 1 + self.mDecay
                p.data.add_(mp, alpha=(0.5 if half else 1) * group["lr"])

                if(args.MH):
                    dist1 = torch.norm(temp2 + temp * (0.5 if half else 1) - p)**2 #q(θ_t+1|θ_t)
                    if(self.lastP[i][j] is not None):
                        dist2 = torch.norm(temp2 + temp * self.lastM - self.lastP[i][j])**2 #q(θ_t|θ_t+1)
                    else:
                        dist2 = torch.zeros(1)
                    self.lastP[i][j] = temp2
                    self.lastM = 0.5 if half else 1
                    dist += (dist2.item() - dist1.item()) #/ p.numel()
            if(args.MH and (args.temperature > 0)):
                self.paramProb += dist / (4 * group["lr"] * args.temperature * args.momentumDecay)

    def getParamProb(self):
        for i, (group, mGroup) in enumerate(zip(self.param_groups, self.mGroups)):
            dist = 0
            for j, (p, mp) in enumerate(zip(group["params"], mGroup)):
                temp = p.grad.data + p / (args.priorSigma ** 2)

                temp = ((1 - self.mDecay) * mp - temp * group["lr"]) / (1 + self.mDecay)
                if(self.lastP[i][j] is not None):
                    dist2 = torch.norm(p + temp * self.lastM - self.lastP[i][j])**2 #q(θ_t|θ_t+1)
                else:
                    dist2 = torch.zeros(1)
                self.lastP[i][j] = None
                self.lastM = 1
                dist += (dist2.item()) #/ p.numel()
            if(args.MH and (args.temperature > 0)):
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
        print("q term: exp({})\t", -paramProb)
        print("prior term: exp({})\t", -priorTerm)
    with torch.no_grad():
        for (data, target) in trainloader:

            data, target = Variable(data), Variable(target)
            data = data.to(cuda_device)
            target = target.to(cuda_device)
            total += target.size(0)

            output = (F.softmax(Mnext(data), dim=1) if outfeature > 1 else Mnext(data))
            if outfeature > 1: Onext.predProb = torch.log(output.data[torch.arange(target.size(0)),target]) / Onext.annealing
            else: Onext.predProb = (output.data - target) ** 2 / 2 / Onext.annealing
            output = (F.softmax(Mcur(data), dim=1) if outfeature > 1 else Mcur(data))
            if outfeature > 1: Ocur.predProb = torch.log(output.data[torch.arange(target.size(0)),target]) / Onext.annealing
            else: Ocur.predProb = (output.data - target) ** 2 / 2 / Onext.annealing

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
    criterion = nn.CrossEntropyLoss().cuda() if outfeature > 1 else nn.MSELoss().cuda()

    train_acc_list = []
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    sampleList = [MLP(sample=True).to(cuda_device) for _ in range(args.sampleNumber)]
    sampleId = 0

    bestModel = MLP().to(cuda_device)
    bestOpt = torch.optim.SGD(bestModel.parameters(), lr=args.lr)
    bestOpt = SGHMC(
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
            group["lr"] = args.lr
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
        line = "Epoch: {}\tLearningRate: {}\tStepTemperature:{}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.temperature)
        print(line)
        log.write(line + "\n")
        log.flush()

        annealTemp = (1 + args.annealing) / 2 + (1 - args.annealing) / 2 * math.cos(epoch * math.pi / args.epochs)
        optimizer.annealing = annealTemp
        bestOpt.annealing = annealTemp

        with tqdm.tqdm(enumerate(trainloader), total=len(trainloader)) as t:
            t.set_description(f"Epoch {epoch} train")
            for batch_idx, (data, target) in t:
                i += 1

                data, target = Variable(data), Variable(target)
                data = data.cuda()
                target = target.cuda()

                output = model(data)
                loss = criterion(output, target)
                if outfeature > 1: prediction = output.data.max(1)[1]
                #print(output,'\n')
                optimizer.zero_grad()
                loss.backward()

                if (False == args.MH):
                    optimizer.step()
                else:
                    if (0 != (batch_idx+1) % args.MHInterval):
                        #optimizer.step(half=(True if (1 == (batch_idx+1) % args.MHInterval) else False))
                        optimizer.step()
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
                        optimizer.step()

                train_loss += loss.item() * target.size(0)
                total += target.size(0)
                if outfeature > 1: correct += prediction.eq(target.data).sum().item()

                # print("accuracy:",accuracy)

                t.set_postfix(
                    {
                        "loss": f"{train_loss/total:.3f}",
                        "acc": f"{100.*correct/total:.3f}%, {correct}/{total}",
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
        val_accuracy, val_loss = evaluate(model, test_loader)
        if (val_accuracy > best_prec):
            best_prec = val_accuracy

        if (False == args.MH):
            line = "Epoch: {}\tTime: {:.1f}s\tTrain Loss: {:.3f}\tTrain Acc: {:.3f}%\nVal Loss: {:.3f}\tVal Acc: {:.3f}\tBest Acc: {:.3f}".format(
                epoch, t,train_loss/total, 100*correct/total, val_loss, val_accuracy.item(), best_prec)
            print(line)
            log.write(line + "\n")
            log.flush()
        else:
            line = "Epoch: {}\tTime: {:.1f}s\tTrain Loss: {:.3f}\t Train Acc: {:.3f}%\nVal Loss: {:.3f}\tVal Acc: {:.3f}\tBest Acc: {:.3f}\tMH Accept Rate: {:.3f}".format(
                epoch, t,train_loss/total, 100*correct/total, val_loss, val_accuracy.item(), best_prec, MH_accept/MH_total)
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
        test_acc_list.append(val_accuracy.item())

    print("Total number of steps: {}".format(i))
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, sampleList


def evaluate(model, test_loader):
    model.eval()
    outputs = []
    accuracies = []
    val_loss = 0
    total = 0
    criterion = nn.CrossEntropyLoss().cuda() if outfeature > 1 else nn.MSELoss().cuda()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            val_loss += criterion(output, target)
            total += 1
            if outfeature > 1:
                prediction = output.data.max(1)[1]
                val_accuracy = torch.mean(prediction.eq(target.data).float()) * 100
            outputs.append(output)
            if outfeature > 1: accuracies.append(val_accuracy)
            else: accuracies.append(0.)

    return torch.mean(torch.tensor(accuracies)), val_loss / total


start_time = time.time()

all_losses = []
all_times = []

for _ in range(1):
    #args.temperature = 1 / 10 ** rerun

    losses = []
    s_time = time.time()
    for _ in range(args.rerun):
        samples = []
        model = MLP().to(cuda_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        optimizer = SGHMC(
            optimizer,
            noise=True,
            temperature=args.temperature,
            datasize=datasize,
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
        if(args.fixed_lr):
            for group in optimizer.param_groups:
                group["lr"] = args.lr if group["lr"] is not None else None
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1e8, gamma=1, last_epoch=-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)
        for epoch in range(args.start_epoch):
            lr_scheduler.step()
        train_loss, train_acc, test_loss, test_acc, sample = train(start_epoch, best_prec, model, trainloader, testloader, optimizer, lr_scheduler, args, cuda_device)
        samples += sample

        criterion = nn.CrossEntropyLoss().cuda() if outfeature > 1 else nn.MSELoss().cuda()
        with tqdm.tqdm(enumerate(testloader), total=len(testloader)) as t:
            t.set_description(f"Sample test")
            fac = 0
            for sample in samples:
                sample.eval()
                fac += sample.predWeight
            correct = 0
            total = 0
            loss = 0
            
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
                    loss += criterion(output / fac, target) * target.size(0)
                    if outfeature > 1:
                        prediction = output.data.max(1)[1]
                        correct += prediction.eq(target.data).sum().item()
                    total += target.size(0)

                    t.set_postfix(
                        {
                            "loss": f"{loss/total:.3f}",
                            "acc": f"{100.*correct/total:.3f}%, {correct}/{total}",
                        }
                    )
            line = "Sample test: loss: {:.3f}, acc: {:.3f}%".format(loss.item()/total, 100*correct/total)
            print(line)
            log.write(line + "\n")
            log.flush()

            losses.append(loss.item()/total)

    all_losses.append(torch.sqrt(torch.tensor(losses)))
    all_times.append(time.time() - s_time)

for t, a in zip(all_times,all_losses):
    line = "time: {:.3f}s\tloss mean: {:.3f}\tloss std: {:.3f}\n".format(t / args.rerun, a.mean(), a.std(unbiased=False))
    print(line)
    log.write(line + "\n\n")
    log.write("==================================================\n")
    log.flush()

line = "Total time: {}s".format(time.time() - start_time)
print(line)
log.write(line + "\n\n")
log.write("==================================================\n")
log.flush()