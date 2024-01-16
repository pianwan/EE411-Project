import abc

import torch
import torchvision.models


class ModelConfig:

    def __init__(self, args):
        if args.model == 'resnet18':
            self.model = ResNet18(args)
        elif args.model == 'mlp':
            self.model = None

        self.model.build_network()
        self.model.setup_optimizer()

    def get_network(self):
        return self.model.network

    def get_optimizer(self):
        return self.model.optimizer


class Model:
    def __init__(self, args):
        self.args = args
        self.optimizer = None
        self.network = None

    @abc.abstractmethod
    def build_network(self):
        pass

    def setup_optimizer(self):
        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(list(self.network.parameters()), lr=self.args.lr,
                                             momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(list(self.network.parameters()), lr=self.args.lr, betas=(0.9, 0.999),
                                              weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "RMSProp":
            self.optimizer = torch.optim.RMSprop(list(self.network.parameters()), lr=self.args.lr)
        else:
            raise AttributeError("select a correct optimizer")


class ResNet18(Model):
    def build_network(self):
        self.network = torchvision.models.resnet18()
        self.network.fc = torch.nn.Linear(self.network.fc.in_features, 10)
