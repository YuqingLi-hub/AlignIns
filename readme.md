## Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection

We provide the code of proposed AlignIns.

## Usage

### Environment

Our code does not rely on special libraries or tools, so it can be easily integrated with most environment settings. 

If you want to use the same settings as us, we provide the conda environment we used in `env.yaml` for your convenience.

### Dataset

CIFAR-10 and CIFAR-100 datasets are available on `torchvision` and will be downloaded automatically.

### Example

Generally, to run a case with default settings, you can easily use the following command:

```
python federated.py \
--poison_frac 0.3 --num_corrupt 4 \
--aggr alignins --data cifar10 --attack badnet
```

If you want to run a case with non-IID settings, you can easily use the following command:

```
python federated.py \
--poison_frac 0.3 --num_corrupt 4 \
--non_iid --alpha 0.5 \
--aggr alignins --data cifar10 --attack badnet
```

Here,

| Argument        | Type       | Description   | Choice |
|-----------------|------------|---------------|--------|
| `aggr`         | str   | Defense method applied by the server | avg, mkrum, flame, rfa, foolsgold, mmetric, rlr, lockdown|
| `data`    |   str     | ID data for all clients          | cifar10, cifar100 |
| `attack`         | str | attack method used   | badnet, pgd, neurotoxin, lie |
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A |
| `alpha`         | float | Data heterogeneous degree     | from 0.1 to 1.0|

For other arguments, you can check the `federated.py` file where the detailed explanation is presented.


