# Asymptotic Scaling Rates

This repository contains experiments demonstrating that model architecture choices change the slopes of L(P) scaling rates, meaning that asymptotically faster scaling is possible. GLUs have a quadratic term, which lets them exactly represent piecewise quadratic functions. 

## Running it

The experiment is managed with hydra configs.

- Running a sweep: `python3 -m divine_scaling.experiment -m model_arch=mlp,glu "n_hidden=range(1,30)"`
- Unit tests: `python3 -m unittest discover -s divine_scaling -p "*_test.py"`


To reference this work, please cite:
```
{paper}
```