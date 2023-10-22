import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class SGHMC(Optimizer):
    def __init__(
        self,
        optim,
        grad_scaling=1.0,
        MH=False,
        temperature=1.0,
        datasize=None,
        mDecay=0.1,
        priorSigma = 100,
        lr = 0.1,
    ):
        assert isinstance(optim, torch.optim.SGD)
        super(SGHMC, self).__init__(optim.param_groups, optim.defaults)
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
                p.grad = torch.zeros(p.shape).cuda()
                self.mGroups[mGroupsIndex].append(torch.randn(p.shape).cuda() * np.sqrt(lr / datasize))
        self.MH = MH
        self.temperature = temperature
        self.datasize = datasize
        self.mDecay = mDecay
        self.priorSigma = priorSigma

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
                d_p = p.grad.data + p / (self.priorSigma ** 2)

                temp2 = p
                mp.data.add_(mp, alpha=-self.mDecay)
                mp.data.add_(d_p, alpha=(- group["lr"]) / self.annealing)
                temp = mp / (1 + self.mDecay)

                eps = torch.randn(p.size()).cuda()
                noise = (
                    group["lr"] * self.temperature * self.mDecay
                ) ** 0.5 * 2 * eps
                mp.data.add_(noise)
                mp /= 1 + self.mDecay
                p.data.add_(mp, alpha=(0.5 if half else 1) * group["lr"])

                if(self.MH):
                    dist1 = torch.norm(temp2 + group["lr"] * temp * (0.5 if half else 1) - p)**2 #q(θ_t+1|θ_t)
                    if(self.lastP[i][j] is not None):
                        dist2 = torch.norm(temp2 + group["lr"] * temp * self.lastM - self.lastP[i][j])**2 #q(θ_t|θ_t+1)
                    else:
                        dist2 = torch.zeros(1)
                    self.lastP[i][j] = temp2
                    self.lastM = 0.5 if half else 1
                    dist += (dist2.item() - dist1.item())
            if(self.MH and (self.temperature > 0)):
                self.paramProb += dist / (4 * group["lr"] * self.temperature * self.mDecay)

    def getParamProb(self):
        for i, (group, mGroup) in enumerate(zip(self.param_groups, self.mGroups)):
            dist = 0
            for j, (p, mp) in enumerate(zip(group["params"], mGroup)):
                temp = p.grad.data + p / (self.priorSigma ** 2)

                temp = ((1 - self.mDecay) * mp - temp * group["lr"]) / (1 + self.mDecay)
                if(self.lastP[i][j] is not None):
                    dist2 = torch.norm(p + temp * self.lastM - self.lastP[i][j])**2 #q(θ_t|θ_t+1)
                else:
                    dist2 = torch.zeros(1)
                self.lastP[i][j] = None
                self.lastM = 1
                dist += (dist2.item())
            if(self.MH and (self.temperature > 0)):
                self.paramProb += dist / (4 * group["lr"] * self.temperature * self.mDecay)
        temp = self.paramProb
        self.paramProb = 0
        return temp

    def getPrior(self):
        temp = 0
        for _, group in enumerate(self.param_groups):
            for _, p in enumerate(group["params"]):
                temp += (torch.norm(p).item() ** 2)
        return temp