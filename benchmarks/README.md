# Accuracy benchmarks of AadpMoE

## Evaluations through lm-evaluation-harness

We recommand using the lm-evaluation-harness library to evaluate the adap-gating mechaism:

```
python run_acc_benchmark.py --task mmlu --size 7 --hessian --threshold 0 0.001 0.003 0.005 0.007
```

## Evaluations through chain-of-thought-hub

However, the accuracy results in this paper was obtained through the chain-of-thought-hub, and the accuracy results is a little bit different from the results obtained from lm-evaluation-harness. You can use the chain-of-thought-hub to fully reproduce the results in the paper.

```
cd chain-of-thought-hub
python run_mmlu.py --hessian --threshold 0 0.001 0.003 0.005 0.007 0.01 0.013

```
