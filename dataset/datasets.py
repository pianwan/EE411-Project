import abc

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets


class DatasetConfig:
    def __init__(self, args):
        self.args = args
        if args.dataset == 'cifar-10':
            self.dataset = CIFAR10Loader(args)
        elif args.dataset == 'cifar-100':
            self.dataset = CIFAR100Loader(args)
        elif args.dataset == 'svhn':
            self.dataset = SVHNLoader(args)
        self.dataset.load_data()
        self.dataset.post_load_data()

    def get_train_dataset(self):
        return self.dataset.train_loader

    def get_test_dataset(self):
        return self.dataset.test_loader


class Dataset:
    def __init__(self, args):
        self.args = args
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @abc.abstractmethod
    def load_data(self):
        pass

    def post_load_data(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )


class CIFAR10Loader(Dataset):
    def load_data(self):
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)


class CIFAR100Loader(Dataset):
    def load_data(self):
        self.train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform)


class SVHNLoader(Dataset):
    def load_data(self):
        self.train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=self.transform)
        self.test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=self.transform)
