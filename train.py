import os
import random

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange

from config import config_parser
from dataset.datasets import DatasetConfig
from metrics import compute_metrics
from model.model import ModelConfig
from utils import save_model, load_model, count_params


def train(args):
    # load data
    print("> Loading data")
    dataset = DatasetConfig(args)
    train_loader, test_loader = dataset.get_train_dataset(), dataset.get_test_dataset()

    # load model
    print("> Loading model")
    model = ModelConfig(args)
    optimizer = model.get_optimizer()
    network = model.get_network().to(args.device)

    # model params
    num_params = count_params(network)
    print(f"> Total number of parameters {num_params}")

    criterion = BCEWithLogitsLoss()
    # Cosine Annealing learning rate
    lr_scheduler = CosineAnnealingLR(optimizer, args.epoch)

    train_acc, train_loss = [], []
    test_acc, test_loss = [], []

    # load weights
    start = 0
    if args.load_weights:
        start, train_acc, train_loss, test_acc, test_loss = load_model(model, args.save_path, args)

    # train
    print("> Start training")
    for epoch in trange(start + 1, args.epoch + 1):
        # train mode
        network.train()
        loss_epoch = 0

        # train for all the data in one epoch
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).to(torch.float32).to(args.device)
            optimizer.zero_grad()
            preds = network(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_epoch += loss.item() / inputs.shape[0]

        tqdm.write(f"Epoch: [{epoch}/{args.epoch}], Loss: {loss_epoch}")
        train_loss.append(loss_epoch)

        # Compute test metrics
        if epoch % args.metrics_iter == 0 or epoch == args.epoch:
            print(f"Computing metrics for epoch {epoch}")
            te_acc, te_loss = compute_metrics(network, criterion, test_loader, args)
            tr_acc, _ = compute_metrics(network, criterion, train_loader, args)
            test_loss.append(te_loss)
            test_acc.append(te_acc)
            train_acc.append(tr_acc)

        # Save model
        if epoch % args.save_iter == 0 or epoch == args.epoch:
            save_model(model, args.save_path, epoch, train_acc, train_loss, test_acc, test_loss)
            print(f"Saved checkpoints for epoch {epoch} at {args.save_path}")


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
    train(args)
