import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # config file
    parser.add_argument('--config', is_config_file=True, default='./configs/sample.txt',
                        help='config file path')

    parser.add_argument("--device", type=str, default="cpu",
                        help='Device to run with')
    parser.add_argument("--model", type=str, default="resnet19",
                        help='model to use')
    parser.add_argument("--dataset", type=str, default="resnet18",
                        help='dataset to use')
    parser.add_argument("--num_classes", type=int, default=10,
                        help='number of classes')
    parser.add_argument("--optimizer", type=str, default="SGD",
                        help='dataset to use')
    parser.add_argument("--seed", type=int, default=0,
                        help='the seed for reproducibility')
    parser.add_argument("--batch_size", type=int, default=512,
                        help='batch size')
    parser.add_argument("--epoch", type=int, default=4000,
                        help='epoch')
    parser.add_argument("--lr", type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help='weight_decay')
    parser.add_argument("--beta", type=int, default=0,
                        help='beta for beta_lasso')
    parser.add_argument("--beta_lambda", type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument("--alpha", type=int, default=1,
                        help='alpha')
    # training params
    parser.add_argument("--load_weights", action='store_true',
                        help='weight ckpt loading')
    parser.add_argument("--weight_iter", type=int, default=100,
                        help='iter of weight to load')
    parser.add_argument("--save_iter", type=int, default=100,
                        help='iter of weight to save')
    parser.add_argument("--metrics_iter", type=int, default=100,
                        help='iter to compute metrics')
    parser.add_argument("--alpha", type=int, default=1,
                        help='base number of channels')
    parser.add_argument("--save_path", type=str, default="./checkpoints",
                        help='save path for checkpoint')
    return parser
