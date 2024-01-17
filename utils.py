import os

import torch


def save_model(model, path, step, losses):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, '{:06d}.ckpt'.format(step))
    torch.save({
        'step': step,
        'network': model.get_network().state_dict(),
        'optimizer': model.get_optimizer().state_dict(),
        'losses': losses,
    }, path)
    print("Saved checkpoints at", path)


def load_model(model, path, args):
    model.build_network()
    path = os.path.join(path, '{:06d}.ckpt'.format(args.weight_iter))
    ckpt = torch.load(path)
    model.get_network().load_state_dict(ckpt["network"])
    model.get_optimizer().load_state_dict(ckpt["optimizer"])
    print("Loaded checkpoint at", path)
    return ckpt["step"], ckpt["losses"]
