import abc

import torch
import torchvision.models

from model.optim import BetaLASSO


class ModelConfig:

    def __init__(self, args):
        if args.model == 'resnet18':
            self.model = ResNet18(args)
        elif args.model == 'mlp':
            self.model = MLP3(args)
        elif args.model == 'mlps':
            self.model = MLPS(args)
        elif args.model == 'sconv':
            self.model = SConv(args)      
        elif args.model == 'slocal':
            self.model = SLocal(args)      

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
        elif self.args.optimizer == "BetaLasso":
            self.optimizer = BetaLASSO(list(self.network.parameters()), lr=self.args.lr, beta=self.args.beta,
                                       lambda_=self.args.beta_lambda)
        else:
            raise AttributeError("select a correct optimizer")


class ResNet18(Model):
    def build_network(self):
        self.network = torchvision.models.resnet18()
        self.network.fc = torch.nn.Linear(self.network.fc.in_features, self.args.num_classes)
        
        
class SConv(Model):
    def build_network(self):
        self.network = nn.Sequential(
            ConvLayer(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Linear(64 * 28 * 28, self.args.num_classes)
        )


class SLocal(Model):
    def build_network(self):
        self.network = LocalConnectLayer(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)


class MLP3(Model):
    def build_network(self):
        self.network = nn.Sequential(
            FullConnectLayer(in_features=28*28, out_features=512),
            FullConnectLayer(in_features=512, out_features=256),
            nn.Linear(256, self.args.num_classes)
        )


class MLPS(Model):
    def build_network(self):
        self.network = nn.Sequential(
            FullConnectLayer(in_features=28*28, out_features=512),
            FullConnectLayer(in_features=512, out_features=256),
            FullConnectLayer(in_features=256, out_features=128),
            nn.Linear(128, self.args.num_classes)
        )
  
               

