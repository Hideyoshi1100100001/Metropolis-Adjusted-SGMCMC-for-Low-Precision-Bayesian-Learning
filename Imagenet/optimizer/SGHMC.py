import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class SGHMC(Optimizer):
    def __init__(
        self,
        optim,
        grad_scaling=1.0,
        MH=False,
        noise=False,
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
        self.quantizeList = []
        self.save_hessian = None
        self.mGroups = []
        mGroupsIndex = -1
        for group in self.param_groups:
            self.mGroups.append([])
            mGroupsIndex += 1
            for p in group["params"]:
                p.grad = torch.zeros(p.shape).cuda()
                self.mGroups[mGroupsIndex].append(torch.randn(p.shape).cuda() * np.sqrt(lr / datasize))
        self.MH = MH
        self.noise = noise
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

    def get_trace(self, p, grad):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        if grad.grad_fn is None:
            raise RuntimeError('Gradient tensor does not have grad_fn. When calling\n' +
                        '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                        '\t\t\t  set to True.')

        v = 2 * torch.randint_like(p, high=2) - 1

        for v_i in v:
            v_i[v_i < 0.] = -1.
            v_i[v_i >= 0.] = 1.

        hv = torch.autograd.grad(
            grad,
            p,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)[0]

        param_size = hv.size()
        if len(param_size) <= 2:  # for 0/1/2D tensor
            # Hessian diagonal block size is 1 here.
            # We use that torch.abs(hv * vi) = hv.abs()
            return hv.abs()

        elif len(param_size) == 4:  # Conv kernel
            # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
            # We use that torch.abs(hv * vi) = hv.abs()
            return torch.mean(hv.abs(), dim=[2, 3], keepdim=True)

    def step(self, lr=None, half=False):
        #print("==================================")
        if self.paramProb is None:
            self.paramProb = 0
        for i, (group, mGroup) in enumerate(zip(self.param_groups, self.mGroups)):
            if lr:
                group["lr"] = lr
            for j, (p, mp) in enumerate(zip(group["params"], mGroup)):
                if self.MH and self.quantizeList[j]:
                    if self.save_hessian is not None:
                        hessian = self.save_hessian
                        self.save_hessian = None
                    else:
                        hessian = torch.norm(self.get_trace(p, p.grad)).item() ** 2

                d_p = p.grad.data + p / (self.priorSigma ** 2)

                temp2 = p
                mp.data.add_(mp, alpha=-self.mDecay)
                mp.data.add_(d_p, alpha=(- group["lr"]) / self.annealing)
                temp = mp / (1 + self.mDecay)

                if self.noise:
                    eps = torch.randn(p.size()).cuda()
                    noise = (
                        group["lr"] * self.temperature * self.mDecay
                    ) ** 0.5 * 2 * eps
                    mp.data.add_(noise)

                mp /= 1 + self.mDecay
                p.data.add_(mp, alpha=(0.5 if half else 1) * group["lr"])

                # calculate M-H transition probability
                if self.MH and self.quantizeList[j]:
                    dist1 = torch.norm(temp2 + group["lr"] * temp * (0.5 if half else 1) - p)**2 #q(θ_t+1|θ_t)
                    if(self.lastP[i][j] is not None):
                        dist2 = torch.norm(temp2 + group["lr"] * temp * self.lastM - self.lastP[i][j])**2 #q(θ_t-1|θ_t)
                    else:
                        dist2 = torch.zeros(1)
                    self.lastP[i][j] = temp2
                    self.lastM = 0.5 if half else 1
                    dist = (dist1.item() - dist2.item())

                    beta_w = temp2.mean((1,2,3)).view(-1,1,1,1)
                    alpha_w = torch.sqrt(((temp2-beta_w)**2).mean((1,2,3))).view(-1,1,1,1)
                    alpha = torch.norm(alpha_w).item() ** 2
                    self.paramProb += dist / (8 * group["lr"] ** 2 * alpha * hessian)

    def getParamProb(self):
        for i, (group, mGroup) in enumerate(zip(self.param_groups, self.mGroups)):
            for j, (p, mp) in enumerate(zip(group["params"], mGroup)):
                if not self.quantizeList[j]:
                    continue

                hessian = torch.norm(self.get_trace(p, p.grad)).item() ** 2
                self.save_hessian = hessian

                temp = p.grad.data + p / (self.priorSigma ** 2)

                temp = ((1 - self.mDecay) * mp - temp * group["lr"]) / (1 + self.mDecay)
                if(self.lastP[i][j] is not None):
                    dist2 = torch.norm(p + temp * self.lastM - self.lastP[i][j])**2 #q(θ_t|θ_t+1)
                else:
                    dist2 = torch.zeros(1)
                self.lastP[i][j] = None
                self.lastM = 1

                beta_w = p.mean((1,2,3)).view(-1,1,1,1)
                alpha_w = torch.sqrt(((p-beta_w)**2).mean((1,2,3))).view(-1,1,1,1)
                alpha = torch.norm(alpha_w).item() ** 2
                self.paramProb -= dist2.item() / (8 * group["lr"] ** 2 * alpha * hessian)

        temp = self.paramProb
        self.paramProb = 0
        return temp

    def getPrior(self):
        temp = 0
        for _, group in enumerate(self.param_groups):
            for _, p in enumerate(group["params"]):
                temp += (torch.norm(p).item() ** 2)
        return temp