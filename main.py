import argparse
import json

from data_loader import get_dataloader
from solver import Solver


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', action='store', type=str, required=True)
    parser.add_argument('--num_epoch', action='store', type=int, default=3000)
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    train_loader = {}
    test_loader = {}
    train_loader['a'], train_loader['b'] = get_dataloader(a=config['class_a'], b=config['class_b'], training=True)
    test_loader['a'], test_loader['b'] = get_dataloader(a=config['class_a'], b=config['class_b'], training=False)

    solver = Solver(train_loader, test_loader, config)
    solver.train(args.num_epoch)