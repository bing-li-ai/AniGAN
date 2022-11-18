import copy
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler
from .funit_model import FUNITModel


class Trainer(nn.Module):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.model = FUNITModel(cfg)
        lr_gen = cfg['lr_gen']
        lr_dis = cfg['lr_dis']
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        self.dis_opt = torch.optim.RMSprop(
            [p for p in dis_params if p.requires_grad],
            lr=lr_gen, weight_decay=cfg['weight_decay'])
        self.gen_opt = torch.optim.RMSprop(
            [p for p in gen_params if p.requires_grad],
            lr=lr_dis, weight_decay=cfg['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, cfg)
        self.gen_scheduler = get_scheduler(self.gen_opt, cfg)

        self.use_encoder_adv = False
        if self.use_encoder_adv:
            dis_c_params = list(self.model.dis_c.parameters())
            self.dis_c_opt = torch.optim.RMSprop(
                [p for p in dis_c_params if p.requires_grad],
                lr=lr_gen, weight_decay=cfg['weight_decay'])
            self.dis_c_scheduler = get_scheduler(self.dis_c_opt, cfg)

        self.apply(weights_init(cfg['init']))
        self.model.gen_test = copy.deepcopy(self.model.gen)

    def load_ckpt(self, ckpt_name):
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict['gen'])
        self.model.gen_test.load_state_dict(state_dict['gen_test'])
        print('----------load success----------')

    def forward(self, *inputs):
        print('Forward function not implemented.')
        pass


def get_scheduler(optimizer, hp, it=-1):
    if 'lr_policy' not in hp or hp['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hp['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hp['step_size'],
                                        gamma=hp['gamma'], last_epoch=it)
    else:
        return NotImplementedError('%s not implemented', hp['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun
