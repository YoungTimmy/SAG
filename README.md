# Mitigating Semantic Confusion from Hostile Neighborhood for Graph Active Learning
Implementation for the **SAG paper** [ “Mitigating Semantic Confusion from Hostile Neighborhood for Graph Active Learning”](https://arxiv.org/abs/2308.08823) in CIKM2023.

## Folders and Files:
    /data: benchmark datasets
    /log: storage of results
    model.py: define a two-layer GCN backbone model
    sag.py: train and evaluate SAG

## Basic Requirements:
    python == 3.7.10
    pytorch == 1.8.0
    CUDA == 11.4
    torch-geometric == 2.1.0
    torch-scatter == 2.0.6
    torch-sparse == 0.6.12
    scipy == 1.5.2
    numpy == 1.19.5
    scikit-learn == 0.22.1

## Run:
To replicate the experiments, please simply run:

    python sag.py

The default dataset is Cora and hyperparameters are listed in the argparser. 

Detailed logs and results can be viewed in the log file in /log.


For Citeseer and Pubmed, please run:

    python sag.py --dataset Citeseer  --lamb 0.5 --theta 0.01
    python sag.py --dataset Pubmed --lamb 0.2 --theta 0.01

respectively.

## Citation

Please cite our paper if you make use of this code in your own work:

```
@inproceedings{yang2023mitigating,
  title={Mitigating Semantic Confusion from Hostile Neighborhood for Graph Active Learning},
  author={Yang, Tianmeng and Zhou, Min and Wang, Yujing and Lin, Zhengjie and Pan, Lujia and Cui, Bin and Tong, Yunhai},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={4380--4384},
  year={2023}
}
```

