import copy
import torch
import torch.nn as nn

from .networks import FewShotGen, GPPatchMcResDis, Content_classifier, GPPatchMcResDis_Switch_FM_up, GPPatchMcResDis_Switch_FM_up_class, GPPatchMcResDis_Switch_full_FM, GPPatchMcResDis_Multi_Switch_FM_up
from .blocks import RhoClipper


class FUNITModel(nn.Module):
    def __init__(self, hp):
        super(FUNITModel, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.gen_test = copy.deepcopy(self.gen)
        self.Rho_clipper = RhoClipper(0, 1)

        self.use_encoder_adv = False
        self.use_attn_encoder = False
        self.use_D_class = False
        self.use_SwitchD = True
        self.use_full_DB = False
        self.use_multi_class = True

        if self.use_SwitchD:
            if self.use_D_class:
                self.dis = GPPatchMcResDis_Switch_FM_up_class(hp['dis'])
                self.lambda_class = 0.2
            else:
                if self.use_multi_class:
                    self.dis = GPPatchMcResDis_Multi_Switch_FM_up(hp['dis'])
                else:
                    if self.use_full_DB:
                        self.dis = GPPatchMcResDis_Switch_full_FM(hp['dis'])
                    else:
                        self.dis = GPPatchMcResDis_Switch_FM_up(hp['dis'])

        else:
            self.dis = GPPatchMcResDis(hp['dis'])

        if self.use_encoder_adv:
            self.dis_c = Content_classifier()
            self.lambda_content_adv = 0.2

    def evaluate_reference(self, test_A, test_B):
        self.eval()
        self.gen_test.eval()
        xa = test_A.cuda()
        xb = test_B.cuda()

        if self.use_attn_encoder:
            c_xa = self.gen_test.enc_content(xa, torch.zeros(test_A.size()[0], 1))
        else:
            c_xa = self.gen_test.enc_content(xa)

        s_xb, sf_xb = self.gen_test.Switch_encode(xb, torch.ones(test_B.size()[0], 1))
        fake_B = self.gen_test.decode(c_xa, s_xb, sf_xb)
        return fake_B
