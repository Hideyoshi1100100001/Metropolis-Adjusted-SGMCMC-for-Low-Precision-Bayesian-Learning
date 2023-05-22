# Metropolis-Adjusted Stochastic Gradient MCMC for Low-Precision Bayesian Learning

The code for [Metropolis-Adjusted Stochastic Gradient MCMC for Low-Precision Bayesian Learning](https://openreview.net/forum?id=MPyZNQtHSI).

## UCI

With yacht.data in ./data/,

To run SGD:

```
python3 UCI.py --epochs 300 --batch-size 65 --lr 1e-2 --temperature 0 --priorSigma 1e4 --SavePath ./ckpt/SGD --rerun 5 --dataPath yacht
```

To run SGHMC:

```
python3 UCI.py --epochs 300 --batch-size 65 --lr 1e-2 --temperature 1e-3 --priorSigma 1e4 --SavePath ./ckpt/SGD --rerun 5 --dataPath yacht
```

To run MH:

```
python3 UCI.py --epochs 300 --batch-size 65 --lr 1e-2 --temperature 1e-3 --priorSigma 1e4 --SavePath ./ckpt/SGD --rerun 5 --dataPath yacht --MH --MHInterval 2 --fixed_lr
```

## CIFAR 10

To run SGD:

```
python3 CIFAR10.py --epochs 400 --end-epoch 100 --temperature 0 --SavePath ./ckpt/SGD100
python3 CIFAR10.py --epochs 160 --temperature 0 --LoadPath ./ckpt/SGD100 --SavePath ./ckpt/SGD160
```

To run SGHMC:

```
python3 CIFAR10.py --epochs 160 --temperature 1e-10 --LoadPath ./ckpt/SGD100 --SavePath ./ckpt/SGHMC160
```

To run MH:

```
python3 CIFAR10.py --epochs 140 --temperature 1e-10 --LoadPath ./ckpt/SGD100 --SavePath ./ckpt/MH140 --MH --MHInterval 10 --MHEpsilon 0.2 --fixed_lr
```

