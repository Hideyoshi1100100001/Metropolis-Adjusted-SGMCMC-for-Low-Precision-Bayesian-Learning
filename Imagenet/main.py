import math
import time
import random
import os
import shutil

import sys
sys.path.append("../../MHimagenet/lib/python3.6/site-packages")

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torchvision import datasets, transforms
import scipy.stats

from prefetch_generator import BackgroundGenerator

from utils.binarylib import AdaBin_Conv2d

from nets.resnet18 import resnet18_1w1a
from nets.resnet34 import resnet34_1w1a
from nets.resnet18fp import resnet18_fp
from optimizer.SGHMC import SGHMC

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

parser = argparse.ArgumentParser(description='ResNets with Metropolis-Hastings for CIFAR-10')

#data

parser.add_argument('-d', '--data', default='../../../LargeData/Large/ImageNet', type=str)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--gpu-id', default='0,1,2,3,4,5,6,7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#optimization

parser.add_argument('--fp', action='store_true', help='full precision')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end-epoch', default=0, type=int, metavar='N',
                    help='manual end epoch number')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--temperature', default=0, type=float, help='factor that multiplies gaussian noise')
parser.add_argument('--annealing', default=1, type=float, help='(0, 1], the lower bound of anneal temperature')
parser.add_argument('--momentumDecay', default=0.1, type=float, metavar='M',
                    help='momentum decay')
parser.add_argument('--priorSigma', '--pS', default=1e2, type=float,
                    metavar='P', help='equivalent to weight decay (default: 1e2 -> 1e-4)')
parser.add_argument('--quantizeDecay', '--qd', default=0, type=float, help='decay the real weight to the quantized (default: 0)')
parser.add_argument('--transitionKernel', '--tk', type=float, default=[1, 1], nargs='+', help='possibility of using different transition kernel, [0]: default, [1] - [0]: flip')
parser.add_argument('--flipPossibility', default=0.05, type=float, help='[0,1], possibility of flipping element wise when transision kernel is random flip')
parser.add_argument('--fixed_lr', action='store_true', help='fix the learning rate')

#checkpoint

parser.add_argument('--LoadPath', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--SavePath', help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--saveInterval',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)

#evaluate

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--sampleNumber', default=10, type=int, help='sample number for ensemble')
parser.add_argument('--sampleInterval', default=1, type=int, help='epochs between two samples')
parser.add_argument('--deepEnsemble', default=1, type=int, help='rerun the training')
parser.add_argument('--ece', action='store_true', help='evaluation of ece')
parser.add_argument('--ecebins', default=15, type=int, help='numbers of ECE bins')

#Metropolist-Hastings

parser.add_argument('--MH', action='store_true', help='MH or pure SGLD')
parser.add_argument('--MHInterval', default=50, type=int, help='iterations between two MH tests')
parser.add_argument('--MHEpsilon', default=2e-1, type=float, help='[0,1], larger->easier to pass MH test')
parser.add_argument('--debug_show_prob', action='store_true', help='show the probability about MH test')

#misc

parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
gpuList = [int(x) for x in args.gpu_id.split(",")]
cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", cuda_device)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.manualSeed)

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

def imageNet():
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    datasize = len(datasets.ImageFolder(traindir).imgs)
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    MH_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    return datasize, train_loader, MH_loader, val_loader

datasize, trainloader, MHloader, testloader = imageNet()
print("datasize: ",datasize)

def Log_UP(epoch, total_epochs):
    K_min, K_max = 1e-3, 1e1 
    # K_min, K_max = 1e-1, 1e1
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / total_epochs * epoch)]).float().cuda()

def Metropolis_Hastings(Ocur, Onext, Mcur, Mnext):
    accept = False
    Onext.dataProb = 0
    paramProb = -5000000
    if Ocur.paramProb is None:
        Ocur.paramProb = -5000000
    paramProb = Onext.getParamProb()
    total = 0
    randomValue = torch.rand(1).item()

    ell = None
    priorTerm = (Onext.getPrior() - Ocur.getPrior()) / (2 * args.priorSigma ** 2) / Onext.annealing
    mu_0 = (math.log(randomValue) - paramProb + priorTerm) / datasize
    if(args.debug_show_prob):
        #print("log_paramProb:", -paramProb / datasize)
        #print("mu_0:", mu_0)
        print("q term:\t", math.exp(paramProb))
        print("prior term:\t", math.exp(-priorTerm))
    with torch.no_grad():
        for (data, target) in MHloader:

            data, target = Variable(data), Variable(target)
            data = data.cuda()
            target = target.cuda()
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
                        print("accept rate:\t", math.exp(paramProb - priorTerm) * torch.exp(datasize * ell.mean()).item())
                    return (ell.mean().item() > mu_0)
    Onext.dataProb = Onext.dataProb.item() / total
    if(Onext.dataProb > mu_0):
        accept = True
    return accept


def train(start_epoch, best_prec, model, train_loader, test_loader, optimizer, lr_scheduler, args):
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
    sampleList = [torch.nn.DataParallel(resnet18_1w1a(sample=True).cuda() if not args.fp else resnet18_fp(sample=True).cuda(), device_ids=gpuList) for _ in range(args.sampleNumber)]
    sampleId = 0

    bestModel = resnet18_1w1a().cuda() if not args.fp else resnet18_fp().cuda()
    bestModel = torch.nn.DataParallel(bestModel, device_ids=gpuList)
    bestOpt = torch.optim.SGD(bestModel.parameters(), lr=args.lr)
    bestOpt = SGHMC(
        optim=bestOpt,
        MH=args.MH,
        temperature=args.temperature,
        datasize=datasize,
        mDecay=args.momentumDecay,
        priorSigma=args.priorSigma,
        lr=args.lr
    )
    for name, _ in bestModel.named_parameters():
        if "layer" in name and "conv" in name and "weight" in name:
            bestOpt.quantizeList.append(True)
        else:
            bestOpt.quantizeList.append(False)
    if(args.LoadPath != ''):
        checkpoint = torch.load(os.path.join(args.LoadPath, "checkpoint.pth"))
        bestModel.load_state_dict(checkpoint['state_dict'])
        bestOpt.load_state_dict(checkpoint['optimizer'])
        for group in bestOpt.param_groups:
            group["lr"] = args.lr if group["lr"] is not None else None
    if(args.fixed_lr):
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
        t5correct = 0
        eceTotal = [0 for _ in range(args.ecebins)]
        ecePred = [0 for _ in range(args.ecebins)]
        eceAcc = [0 for _ in range(args.ecebins)]
        ece = 0
        line = "Epoch: {}\tLearningRate: {}\tStepTemperature:{}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.temperature)
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

        with tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) as t:
            t.set_description(f"Epoch {epoch} train")
            for batch_idx, (data, target) in t:
                i += 1

                data, target = Variable(data), Variable(target)
                data = data.cuda()
                target = target.cuda()

                output = model(data)
                loss = criterion(output, target)
                prediction = output.data.max(1)[1]
                t5pred = torch.topk(output.data, k=5, dim=1, largest=True)[1]
                #print(output,'\n')
                optimizer.zero_grad()
                loss.backward(create_graph=True)

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
                t5correct += t5pred.eq(target.view(-1, 1).data).sum().item()

                #calculate ECE

                ece = 0
                if args.ece:
                    predVal = torch.gather(F.softmax(output, dim=1), 1, target.view(-1, 1))
                    for ecei in range(args.ecebins):
                        predPos = (predVal >= eceBin[ecei]) * (predVal < eceBin[ecei + 1])
                        eceTotal[ecei] += predPos.sum().item()
                        ecePred[ecei] += (predVal * predPos).sum().item()
                        eceAcc[ecei] += (prediction.eq(target.data).view(-1, 1) * predPos).sum().item()
                        ece += abs(ecePred[ecei] - eceAcc[ecei])

                t.set_postfix(
                    {
                        "loss": f"{train_loss/total:.3f}",
                        "acc": f"{100.*correct/total:.3f}%",
                        "top5 acc": f"{100.*t5correct/total:.3f}%",
                        "ECE": f"{ece/total:.3f}",
                        "accept": f"{(MH_accept/MH_total if MH_total > 0 else 0):.3f}"
                    }
                )

        lr_scheduler.step()
        Blr_scheduler.step()
        t = time.time() - tic

        if (0 == (epoch + 1) % args.sampleInterval):
            sampleId = (epoch // args.sampleInterval) % args.sampleNumber
            torch.save(model.state_dict(), os.path.join(args.SavePath, "sample_{:d}.pth".format(sampleId)))
            sampleList[sampleId].load_state_dict(torch.load(os.path.join(args.SavePath, "sample_{:d}.pth".format(sampleId))))
            sampleList[sampleId].sampling = True
            #sampleList[sampleId].predWeight = optimizer.param_groups[0]["lr"]
            sampleList[sampleId].predWeight = 0.1

        # validate
        if 0:
            for p in model.parameters():
                print(p[0])
                break
        val_accuracy, val_t5acc, val_loss, val_ece = evaluate(model, test_loader, epoch)
        if (val_accuracy > best_prec):
            best_prec = val_accuracy

        if (False == args.MH):
            line = "Epoch: {}\tTime: {:.1f}s\tTrain Loss: {:.3f}\tTrain Acc: {:.3f}%\tTrain ECE: {:.3f}\nVal Loss: {:.3f}\tVal Acc: {:.3f}\tVal top5 Acc: {:.3f}\tVal ECE: {:.3f}\tBest Acc: {:.3f}".format(
                epoch, t,train_loss/total, 100*correct/total, ece/total, val_loss, val_accuracy, val_t5acc, val_ece, best_prec)
            print(line)
            log.write(line + "\n")
            log.flush()
        else:
            line = "Epoch: {}\tTime: {:.1f}s\tTrain Loss: {:.3f}\t Train Acc: {:.3f}%\tTrain ECE: {:.3f}\nVal Loss: {:.3f}\tVal Acc: {:.3f}\tVal top5 Acc: {:.3f}\tVal ECE: {:.3f}\tBest Acc: {:.3f}\tMH Accept Rate: {:.3f}".format(
                epoch, t,train_loss/total, 100*correct/total, ece/total, val_loss, val_accuracy, val_t5acc, val_ece, best_prec, MH_accept/MH_total)
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
    outputs = []
    val_acc = 0
    t5acc = 0
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
            t5pred = torch.topk(output.data, k=5, dim=1, largest=True)[1]

            #calculate ECE
            if args.ece:
                predVal = torch.gather(F.softmax(output, dim=1), 1, target.view(-1, 1))
                for ecei in range(args.ecebins):
                    predPos = (predVal >= eceBin[ecei]) * (predVal < eceBin[ecei + 1])
                    eceTotal[ecei] += predPos.sum().item()
                    ecePred[ecei] += (predVal * predPos).sum().item()
                    eceAcc[ecei] += (prediction.eq(target.data).view(-1, 1) * predPos).sum().item()
                    ece += abs(ecePred[ecei] - eceAcc[ecei])

            val_acc += prediction.eq(target.data).sum().item()
            t5acc += t5pred.eq(target.view(-1, 1).data).sum().item()
            outputs.append(output)

    if args.ece and epoch == args.end_epoch - 1:
        for ecei in range(args.ecebins):
            if eceTotal[ecei] > 0:
                line = "ECE bin [{:.3f}, {:.3f}], pred / acc / total: {:.3f} / {:.3f} / {}".format(eceBin[ecei], eceBin[ecei + 1], ecePred[ecei] / eceTotal[ecei],
                                                                                            eceAcc[ecei] / eceTotal[ecei], eceTotal[ecei])
            else:
                line = "ECE bin [{:.3f}, {:.3f}], pred / acc / total: 0 / 0 / 0".format(eceBin[ecei], eceBin[ecei + 1])
            print(line)
            log.write(line + "\n")
            log.flush()

    return 100. * val_acc / total, 100. * t5acc / total, val_loss / total, ece / total

samples = []

start_time = time.time()

for _ in range(args.deepEnsemble):
    if args.evaluate:
        samples = [torch.nn.DataParallel(resnet18_1w1a(sample=True).cuda() if not args.fp else resnet18_fp(sample=True).cuda(), device_ids=gpuList) for _ in range(args.sampleNumber)]
        for i in range(args.sampleNumber):
            samples[i].load_state_dict(torch.load(os.path.join(args.SavePath, "sample_{:d}.pth".format(i))))
        break
    model = resnet18_1w1a().cuda() if not args.fp else resnet18_fp().cuda()
    model = torch.nn.DataParallel(model, device_ids=gpuList)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = SGHMC(
        optim=optimizer,
        MH=args.MH,
        temperature=args.temperature,
        datasize=datasize,
        mDecay=args.momentumDecay,
        priorSigma=args.priorSigma,
        lr=args.lr
    )
    for name, _ in model.named_parameters():
        if "layer" in name and "conv" in name and "weight" in name:
            optimizer.quantizeList.append(True)
        else:
            optimizer.quantizeList.append(False)
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
        for group in optimizer.param_groups:
            group["lr"] = args.lr if group["lr"] is not None else None
    if(args.fixed_lr):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1e8, gamma=1, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)
    for epoch in range(args.start_epoch):
        lr_scheduler.step()
    train_loss, train_acc, test_loss, test_acc, sample = train(start_epoch, best_prec, model, trainloader, testloader, optimizer, lr_scheduler, args)
    samples += sample

eceBin = [i * 1.0001 / args.ecebins for i in range(args.ecebins + 1)]
eceTotal = [0 for _ in range(args.ecebins)]
ecePred = [0 for _ in range(args.ecebins)]
eceAcc = [0 for _ in range(args.ecebins)]
ece = 0

#_, _ = evaluate(samples[0], testloader)
with tqdm.tqdm(enumerate(testloader), total=len(testloader)) as t:
    t.set_description(f"Sample test")
    fac = 0
    for sample in samples:
        sample.eval()
        fac += 0.1
    correct = 0
    t5correct = 0
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
            loss -= criterion(torch.log(output), target) * target.size(0)
            prediction = output.data.max(1)[1]
            t5pred = torch.topk(output.data, k=5, dim=1, largest=True)[1]
            correct += prediction.eq(target.data).sum().item()
            t5correct += t5pred.eq(target.view(-1, 1).data).sum().item()
            total += target.size(0)

            #calculate ECE

            if args.ece:
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
                    "top5 acc": f"{100.*t5correct/total:.3f}%, {correct}/{total}",
                    "ece": f"{ece/total:.3f}",
                }
            )
    line = "Sample test: loss: {:.3f}, acc: {:.3f}%, top5 acc: {:.3f}%, ece: {:.3f}".format(loss.item()/total, 100.*correct/total, 100.*t5correct/total, ece/total)
    print(line)
    log.write(line + "\n")
    log.flush()
    if args.ece:
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