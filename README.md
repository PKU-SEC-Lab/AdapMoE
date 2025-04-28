# AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2504.05897-b31b1b.svg)](https://arxiv.org/abs/2504.05897)&nbsp;
</div>

## Introduction

AdapMoE is an algorithm-system co-design framework for efficient Mixture-of-Experts (MoE) inference on edge devices. It addresses the high latency overhead associated with on-demand expert loading through three key innovations:

- **Adaptive Sensitivity-based Expert Gating**: Dynamically adjusts the number of activated experts per layer based on token and layer sensitivity, reducing expert activations by 25% without accuracy loss.

- **Adaptive Expert Prefetching**: Leverages activation similarities across layers to prefetch experts with high accuracy.

- **Dynamic Cache Allocation**: Optimizes GPU cache allocation across layers using a DP-based formulation.


## Installation
```
conda create -n adapmoe python=3.10
conda activate adapmoe
pip install -r requirements.txt
```

## Usage

### Performance Evaluation
#### Basic Inference

```
python run.py --size {#cached experts} --adapgate
```

#### Arguments
| Argument     | Description                       | Default |
| ------------ | --------------------------------- | ------- |
| `--size`     | Number of experts to cache in GPU | 64     |
| `--adapgate` | Enable adaptive gating            | False   |

### Accuracy Evaluation

#### Evaluations through lm-evaluation-harness

We recommand using the lm-evaluation-harness library to evaluate the adap-gating mechaism:

```
cd benchmarks
python run_acc_benchmark.py --task mmlu --size 7 --hessian --threshold 0 0.001 0.003 0.005 0.007
```


| Argument     | Description                       | Default |
| ------------ | --------------------------------- | ------- |
| `--size`     | Mixtral 8x7b or Mixtral 8x22b     | 7     |
| `--task`     | Dataset                           | mmlu   |
| `--hessian`  | Naive or sensitivity-based adaptive gating                           |    |

#### Evaluations through chain-of-thought-hub

However, the accuracy results in this paper was obtained through the chain-of-thought-hub, and the accuracy results is a little bit different from the results obtained from lm-evaluation-harness. You can use the chain-of-thought-hub to fully reproduce the results in the paper.

```
cd cd benchmarks/chain-of-thought-hub
python run_mmlu.py --hessian --threshold 0 0.001 0.003 0.005 0.007 0.01 0.013
```


## Future works

- We have supported CPU-assisted computation and more MoE models in [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE).

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@inproceedings{10.1145/3676536.3676741,
author = {Zhong, Shuzhang and Liang, Ling and Wang, Yuan and Wang, Runsheng and Huang, Ru and Li, Meng},
title = {AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference},
year = {2025},
isbn = {9798400710773},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3676536.3676741},
doi = {10.1145/3676536.3676741},
booktitle = {Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
articleno = {51},
numpages = {9},
series = {ICCAD '24}
}