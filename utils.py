import os

import torch


def save_model(model, path, step, train_acc, train_loss, test_acc, test_loss):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, '{:06d}.ckpt'.format(step))
    torch.save({
        'step': step,
        'network': model.get_network().state_dict(),
        'optimizer': model.get_optimizer().state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }, path)


def load_model(model, path, args):
    model.build_network()
    path = os.path.join(path, '{:06d}.ckpt'.format(args.weight_iter))
    ckpt = torch.load(path)
    model.get_network().load_state_dict(ckpt["network"])
    model.get_optimizer().load_state_dict(ckpt["optimizer"])
    print("Loaded checkpoint at", path)
    return ckpt["step"], ckpt["train_acc"], ckpt["train_loss"], ckpt["test_acc"], ckpt["test_loss"]
