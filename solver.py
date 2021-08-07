import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from model.discriminator import Discriminator
from model.generator import Generator
from model.siamese import Siamese


def split(tensor):
    return torch.split(tensor, 128, dim=-1)

def combine(tensors):
    return torch.cat(tensors, dim=-1)


class Solver():

    def __init__(self, train_loader, test_loader, config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = train_loader
        self.test_loader = test_loader

        # epoch
        self.epoch = 1

        # networks
        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)
        self.siam = Siamese(latent_dim=128).to(self.device)

        # losses log
        self.losses = {
            'gen': [],
            'dis': [],
        }

        # train optimizers
        self.gen_lr = config['optimizers']['gen_lr']
        self.dis_lr = config['optimizers']['dis_lr']
        self.beta1 = config['optimizers']['beta1']
        self.beta2 = config['optimizers']['beta2']
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), self.gen_lr, [self.beta1, self.beta2])
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), self.dis_lr, [self.beta1, self.beta2])

        # epoch save
        self.epoch_save = config['epoch_save']

        # load checkpoint
        if config['resume'] != '':
            checkpoint = torch.load(config['resume'])
            self.epoch = checkpoint['epoch'] + 1
            self.gen.load_state_dict(checkpoint['gen'])
            self.dis.load_state_dict(checkpoint['dis'])
            self.siam.load_state_dict(checkpoint['siam'])
            self.losses = checkpoint['losses']
        
        # losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def reset_grad(self):
        self.dis_opt.zero_grad()
        self.gen_opt.zero_grad()

    def train_step(self, idx, a, b):
        return None, None

    def train(self, num_epoch=3000):

        # loop epoch
        while self.epoch <= num_epoch:

            print('Epoch {}'.format(self.epoch))

            losses_gen = []
            losses_dis = []

            # loop batch
            for idx, (a, b) in tqdm(enumerate(zip(self.train_loader['a'], self.train_loader['b'])), total=len(self.train_loader['a'])):
                if a.shape[0] != b.shape[0]:
                    continue
                loss_gen, loss_dis = self.train_step(idx, a, b)
                losses_gen.append(loss_gen)
                losses_dis.append(loss_dis)

            self.losses['gen'].append(np.mean(filter(None, losses_gen)))
            self.losses['dis'].append(np.mean(filter(None, losses_dis)))

            # # save checkpoint
            # if self.epoch % self.epoch_save == 0:
            #     torch.save({
            #         'epoch': self.epoch,
            #         'gen': self.gen.state_dict(),
            #         'dis': self.dis.state_dict(),
            #         'siam': self.siam.state_dict(),
            #         'losses': self.losses
            #     }, f'./checkpoints/checkpoint_{self.epoch}.pt')

            self.epoch += 1
