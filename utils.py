import os

import torch


def save_model(model, path, step):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, '{:06d}.ckpt'.format(step))
    torch.save({
        'step': step,
        'network': model.network.state_dict(),
        'optimizer': model.optimizer.state_dict(),
    }, path)
    print("Saved checkpoints at", path)


def load_model(model, path, args):
    model.build_network()
    path = os.path.join(path, '{:06d}.ckpt'.format(args.weight_iter))
    ckpt = torch.load(path)
    model.network.load_state_dict(ckpt["network"])
    model.optimizer.load_state_dict(ckpt["optimizer"])
    print("Loaded checkpoint at", path)
    return ckpt["step"]
