# EE411-Project

[EPFL EE411](https://edu.epfl.ch/coursebook/en/fundamentals-of-inference-and-learning-EE-411) Reproducibility Challenge: Towards learning convolutions from scratch

## Team members

- Pian Wan, pian.wan@epfl.ch
- Yuheng Lu, yuheng.lu@epfl.ch
- Xinwei Li, xinwei.li@epfl.ch
- Jiaxing Dong, jiaxin.dong@epfl.ch
- Xinyi Han, xinyi.han@epfl.ch

## Quickstart

### Requirements

Python 3.10

```shell
git clone --recursive git@github.com:pianwan/EE411-Project.git
cd EE411-Project
pip install -r requirements.txt
```

### Datasets

The datasets can be downloaded directly from `torchvision` datasets. We test our models on `CIFAR-10`, `CIFAR-100`, and `SVHN` datasets.

### Run

```shell
python train.py --config ./configs/sample.txt
```
You can specify the config by using the parameter like `--alpha 100 --epoch 4000`.

## Result

TODO

## Project Organization

TODO

## Contribution Details

TODO

## Acknowledgements

We thank Neyshabur Behnam for his work _Towards learning convolutions from scratch_. For reference, you can cite his paper and this repository by:

```tex
@article{neyshabur2020towards,
  title={Towards learning convolutions from scratch},
  author={Neyshabur, Behnam},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={8078--8088},
  year={2020}
}
```

```tex
@misc{wan2024unoffical,
    title={Unofficial implements of Towards learning convolutions from scratch},
    author= {Wan, Pian and Lu, Yuheng and Li, Xinwei and Jiaxing, Dong and Xinyi, Han},
    year={2024},
    note={https://github.com/pianwan/EE411-Project}
}
```