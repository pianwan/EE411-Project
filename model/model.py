import abc

import torchvision.models

from model.components import *
from model.optim import BetaLASSO


class ModelConfig:
    def __init__(self, args):
        if args.model == 'resnet18':
            self.model = ResNet18(args)
        elif args.model == 'mlp3fc':
            self.model = MLP3FC(args)
        elif args.model == 'mlpsfc':
            self.model = MLPSFC(args)
        elif args.model == 'mlpdfc':
            self.model = MLPDFC(args)
        elif args.model == 'sconv':
            self.model = SConv(args)
        elif args.model == 'slocal':
            self.model = SLocal(args)
        elif args.model == 'dconv':
            self.model = DConv(args)
        elif args.model == 'dlocal':
            self.model = DLocal(args)

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


class MLP3FC(Model):
    def build_network(self):
        self.network = nn.Sequential(
            FullConnectLayer(3072, 256),
            FullConnectLayer(256, 256),
            nn.Linear(256, self.args.num_classes)
        )


class MLPSFC(Model):
    def build_network(self):
        self.network = nn.Sequential(
            FullConnectLayer(3072, 768),
            FullConnectLayer(768, 24),
            nn.Linear(24, self.args.num_classes)
        )


class SConv(Model):
    def build_network(self):
        self.network = torch.nn.Sequential(
            ConvLayer(in_channels=3, out_channels=1*self.args.alpha, kernel_size=9, stride=2, padding=0),
            FullConnectLayer(in_features=144*self.args.alpha, out_features=24*self.args.alpha),
            torch.nn.Linear(24*self.args.alpha, self.args.num_classes)
        )


class SLocal(Model):
    def build_network(self):
        self.network = torch.nn.Sequential(
            LocalConnectLayer(in_channels=3, out_channels=1*self.args.alpha, kernel_size=9, stride=2, padding=0, bias=False),
            FullConnectLayer(in_features=144*self.args.alpha, out_features=24*self.args.alpha),
            torch.nn.Linear(24*self.args.alpha, self.args.num_classes)
        )


class DConv(Model):
    def build_network(self):
        self.network = torch.nn.Sequential(
            ConvLayer(in_channels=3, out_channels=1*self.args.alpha, kernel_size=3, stride=1, padding=0),
            ConvLayer(in_channels=1*self.args.alpha, out_channels=2*self.args.alpha, kernel_size=3, stride=2, padding=0),
            ConvLayer(in_channels=2*self.args.alpha, out_channels=2*self.args.alpha, kernel_size=3, stride=1, padding=0),
            ConvLayer(in_channels=2*self.args.alpha, out_channels=4*self.args.alpha, kernel_size=3, stride=2, padding=0),
            ConvLayer(in_channels=4*self.args.alpha, out_channels=4*self.args.alpha, kernel_size=3, stride=1, padding=0),
            ConvLayer(in_channels=4*self.args.alpha, out_channels=8*self.args.alpha, kernel_size=3, stride=2, padding=0),
            ConvLayer(in_channels=8*self.args.alpha, out_channels=8*self.args.alpha, kernel_size=3, stride=1, padding=0),
            ConvLayer(in_channels=8*self.args.alpha, out_channels=16*self.args.alpha, kernel_size=3, stride=2, padding=0),
            FullConnectLayer(in_features=144*self.args.alpha, out_features=64*self.args.alpha),
            torch.nn.Linear(in_features=64*self.args.alpha, out_features=self.args.num_classes)
        )


class DLocal(Model):
    def build_network(self):
        self.network = torch.nn.Sequential(
            LocalConnectLayer(in_channels=3, out_channels=1*self.args.alpha, kernel_size=3, stride=1, padding=0, bias=False),
            LocalConnectLayer(in_channels=1*self.args.alpha, out_channels=2*self.args.alpha, kernel_size=3, stride=2, padding=0, bias=False),
            LocalConnectLayer(in_channels=2*self.args.alpha, out_channels=2*self.args.alpha, kernel_size=3, stride=1, padding=0, bias=False),
            LocalConnectLayer(in_channels=2*self.args.alpha, out_channels=4*self.args.alpha, kernel_size=3, stride=2, padding=0, bias=False),
            LocalConnectLayer(in_channels=4*self.args.alpha, out_channels=4*self.args.alpha, kernel_size=3, stride=1, padding=0, bias=False),
            LocalConnectLayer(in_channels=4*self.args.alpha, out_channels=8*self.args.alpha, kernel_size=3, stride=2, padding=0, bias=False),
            LocalConnectLayer(in_channels=8*self.args.alpha, out_channels=8*self.args.alpha, kernel_size=3, stride=1, padding=0, bias=False),
            LocalConnectLayer(in_channels=8*self.args.alpha, out_channels=16*self.args.alpha, kernel_size=3, stride=2, padding=0, bias=False),
            FullConnectLayer(in_features=144*self.args.alpha, out_features=24*self.args.alpha),
            torch.nn.Linear(24, self.args.num_classes)
        )


class MLPDFC(Model):
    def build_network(self):
        self.network = nn.Sequential(
            FullConnectLayer(3072, 1024),
            FullConnectLayer(1024, 512),
            FullConnectLayer(512, 512),
            FullConnectLayer(512, 256),
            FullConnectLayer(256, 256),
            FullConnectLayer(256, 128),
            FullConnectLayer(128, 128),
            FullConnectLayer(128, 64),
            nn.Linear(64, self.args.num_classes)
        )
