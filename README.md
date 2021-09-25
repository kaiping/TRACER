# TRACER

This repository shares the source code for the paper:

[TRACER: A Framework for Facilitating Accurate and Interpretable Analytics for High Stakes Applications](https://dl.acm.org/doi/10.1145/3318464.3389720)

TRACER is a general framework to facilitate accurate and interpretable predictions,
with a novel model TITV devised for healthcare analytics and other high stakes applications such as financial investment and risk management.

Please refer to ```example_run_script.py``` for the usage of the source code.

## Input Dataset Format:
A pickle file which includes a list of samples with features and labels (i.e., ground truth).  Specifically:

```python
input_dataset = [(x1, y1), (x2, y2), ..., (xn, yn)]
x = {ndarray: {time_window_size, feature_size}}
y = {int} -> 0: negative or 1: positive
```

## Requirements

```
torch==1.9.1
numpy==1.21.2
scipy==1.7.1
scikit-learn==0.24.2
```

## Reference
Kaiping Zheng, Shaofeng Cai, Horng Ruey Chua, Wei Wang, Kee Yuan Ngiam, Beng Chin Ooi.  
**TRACER: A Framework for Facilitating Accurate and Interpretable Analytics for High Stakes Applications.**  
*Proceedings of the 2020 International Conference on Management of Data, SIGMOD Conference 2020, Portland, OR, USA, June 14 - 19, 2020.* 
