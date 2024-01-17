import os
import random

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import config_parser
from dataset.datasets import DatasetConfig
from model.model import ModelConfig
from utils import save_model, load_model


def train(args, device):
    # load data
    print("> Loading data")
    dataset = DatasetConfig(args)
    train_loader, test_loader = dataset.get_train_dataset(), dataset.get_test_dataset()

    # load model
    print("> Loading model")
    model = ModelConfig(args)
    optimizer = model.get_optimizer()
    network = model.get_network().to(device)

    criterion = BCEWithLogitsLoss()
    # Cosine Annealing learning rate
    lr_scheduler = CosineAnnealingLR(optimizer, args.epoch)

    # load weights
    start = 0
    if args.load_weights:
        start = load_model(model, args.save_path, args)

    # train
    print("> Start training")
    for epoch in tqdm(range(start, args.epoch)):
        loss_eopch = 0
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).to(torch.float32).to(device)
            optimizer.zero_grad()
            preds = network(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_eopch += loss.item() / inputs.shape[0]

    tqdm.write(f"Epoch: [{epoch}/{len(train_loader)}], Loss{loss_eopch}")
    if epoch == args.epoch - 1:
        save_model(model, args.save_path, epoch)


if __name__ == '__main__':
    # load config
    print("Loading config")
    parser = config_parser()
    args = parser.parse_args()

    # setup seed (for exp)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.random.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # setup device
    print(f"Use device: {args.device}")

    # train
    print("Start training...")
    train(args, args.device)
