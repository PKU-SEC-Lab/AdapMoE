# AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference

## Installation
```
pip install -r requirements.txt
```

## Running
```
python run.py --size {#cached experts} --adapgate
```

## Future works

- Other utilities in AdapMoE, such the quantization scripts, the hessian scripts and the benchmark scripts will be organized and released.

- We are migrating the AdapMoE to the ktransformers framework to support more MoE models, quantization kernels and CPU computation.
