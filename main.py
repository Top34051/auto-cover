import numpy as np
import json

from data_loader import get_dataloader
from solver import Solver


if __name__ == '__main__':

    train_loader = {}
    test_loader = {}
    train_loader['a'], train_loader['b'] = get_dataloader(a='city-pop', b='jazz', training=True)
    test_loader['a'], test_loader['b'] = get_dataloader(a='city-pop', b='jazz', training=False)

    config = json.load(open('config.json'))

    solver = Solver(train_loader, test_loader, config)

    solver.train(num_epoch=5)