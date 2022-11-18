import numpy as np

from .blocks import *


class GPPatchMcResDis(nn.Module):

    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        cnn_f = [Conv2dBlock(3, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, hp['num_classes'], 1, 1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        index = torch.LongTensor(range(out.size(0))).cuda()
        out = out[index, y, :, :]
        return out, feat


class GPPatchMcResDis_Switch_full_FM(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis_Switch_full_FM, self).__init__()

        #128*128*3 to 128*128*64
        self.conv_1_A = Conv2dBlock(3, 64, 7, 1, 3, pad_type='reflect', norm='none', activation='none')
        self.conv_1_B = Conv2dBlock(3, 64, 7, 1, 3, pad_type='reflect', norm='none', activation='none')

        #128*128*64 to 64*64*128
        conv_2_A = [ActFirstResBlock(64, 64, None, 'lrelu', 'none'),
                  ActFirstResBlock(64, 128, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]
        
        conv_2_B = [ActFirstResBlock(64, 64, None, 'lrelu', 'none'),
                  ActFirstResBlock(64, 128, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        #64*64*128 to 32*32*256
        conv_3_A = [ActFirstResBlock(128, 128, None, 'lrelu', 'none'),
                  ActFirstResBlock(128, 256, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]
        
        conv_3_B = [ActFirstResBlock(128, 128, None, 'lrelu', 'none'),
                  ActFirstResBlock(128, 256, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        #32*32*256 to 16*16*512
        conv_4_A = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                  ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2),
                  nn.LeakyReLU(0.2)]

        conv_4_B = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                    ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                    nn.ReflectionPad2d(1),
                    nn.AvgPool2d(kernel_size=3, stride=2),
                    nn.LeakyReLU(0.2)]

        #16*16*512 to 8*8*1024
        self.conv_A = Conv2dBlock(512, 1024, 4, 2, 1, norm='none', activation='lrelu', activation_first=False)
        self.conv_A_out = Conv2dBlock(1024, 1, 3, 1, 1, norm='none', activation='none', use_bias=False, activation_first=False)

        #16*16*512 to 8*8*1024
        self.conv_B = Conv2dBlock(512, 1024, 4, 2, 1, norm='none', activation='lrelu', activation_first=False)
        self.conv_B_out = Conv2dBlock(1024, 1, 3, 1, 1, norm='none', activation='none', use_bias=False, activation_first=False)

        self.conv_2_A = nn.Sequential(*conv_2_A)
        self.conv_2_B = nn.Sequential(*conv_2_B)
        self.conv_3_A = nn.Sequential(*conv_3_A)
        self.conv_3_B = nn.Sequential(*conv_3_B)
        self.conv_4_A = nn.Sequential(*conv_4_A)
        self.conv_4_B = nn.Sequential(*conv_4_B)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        features = []
        
        current_A = self.conv_1_A(x)
        current_B = self.conv_1_B(x)
        current_A = self.conv_2_A(current_A)
        current_B = self.conv_2_B(current_B)
        
        current = self.feature_switch(current_A, current_B, y)
        features.append(current)

        current_A = self.conv_3_A(current)
        current_B = self.conv_3_B(current)

        current = self.feature_switch(current_A, current_B, y)
        features.append(current)
        
        feat_A = self.conv_4_A(current)
        feat_B = self.conv_4_B(current)

        out_A = self.conv_A(feat_A)
        out_B = self.conv_B(feat_B)

        cFeat = self.feature_switch(out_A, out_B, y)
        features.append(cFeat)

        out_A = self.conv_A_out(out_A)
        out_B = self.conv_B_out(out_B)

        out = self.feature_switch(out_A, out_B, y)
        return out, features

    def feature_switch(self, feature_1, feature_2, label):
        new_label = []
        for y_binary in label:
            if int(y_binary) == 1:
                new_label.append(torch.ones(feature_1.size()[1:]).unsqueeze(0))
            else:
                new_label.append(torch.zeros(feature_1.size()[1:]).unsqueeze(0))

        new_label = torch.cat(new_label, 0).cuda()
        out = torch.mul(new_label, feature_1) + torch.mul((1 - new_label), feature_2)
        return out


class GPPatchMcResDis_Switch_FM_up(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis_Switch_FM_up, self).__init__()

        #128*128*3 to 128*128*64
        self.conv_1 = Conv2dBlock(3, 64, 7, 1, 3, pad_type='reflect', norm='none', activation='none')

        #128*128*64 to 64*64*128
        conv_2 = [ActFirstResBlock(64, 64, None, 'lrelu', 'none'),
                  ActFirstResBlock(64, 128, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        #64*64*128 to 32*32*256
        conv_3 = [ActFirstResBlock(128, 128, None, 'lrelu', 'none'),
                  ActFirstResBlock(128, 256, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        #32*32*256 to 16*16*512
        conv_4_A = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                  ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2),
                  nn.LeakyReLU(0.2)]

        conv_4_B = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                    ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                    nn.ReflectionPad2d(1),
                    nn.AvgPool2d(kernel_size=3, stride=2),
                    nn.LeakyReLU(0.2)]

        #16*16*512 to 8*8*1024
        self.conv_A = Conv2dBlock(512, 1024, 4, 2, 1, norm='none', activation='lrelu', activation_first=False)
        self.conv_A_out = Conv2dBlock(1024, 1, 3, 1, 1, norm='none', activation='none', use_bias=False, activation_first=False)

        #16*16*512 to 8*8*1024
        self.conv_B = Conv2dBlock(512, 1024, 4, 2, 1, norm='none', activation='lrelu', activation_first=False)
        self.conv_B_out = Conv2dBlock(1024, 1, 3, 1, 1, norm='none', activation='none', use_bias=False, activation_first=False)

        self.conv_2 = nn.Sequential(*conv_2)
        self.conv_3 = nn.Sequential(*conv_3)
        self.conv_4_A = nn.Sequential(*conv_4_A)
        self.conv_4_B = nn.Sequential(*conv_4_B)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        features = []
        current = self.conv_1(x)
        current = self.conv_2(current)

        self.use_FM3 = True
        self.use_cFM = True

        if self.use_FM3:
            features.append(current)

        current = self.conv_3(current)

        if self.use_FM3:
            features.append(current)

        feat_A = self.conv_4_A(current)
        feat_B = self.conv_4_B(current)

        out_A = self.conv_A(feat_A)
        out_B = self.conv_B(feat_B)

        if self.use_cFM:
            cFeat = self.feature_switch(out_A, out_B, y)
            features.append(cFeat)

        out_A = self.conv_A_out(out_A)
        out_B = self.conv_B_out(out_B)

        out = self.feature_switch(out_A, out_B, y)
        return out, features

    def feature_switch(self, feature_1, feature_2, label):
        new_label = []
        for y_binary in label:
            if int(y_binary) == 1:
                new_label.append(torch.ones(feature_1.size()[1:]).unsqueeze(0))
            else:
                new_label.append(torch.zeros(feature_1.size()[1:]).unsqueeze(0))

        new_label = torch.cat(new_label, 0).cuda()
        out = torch.mul(new_label, feature_1) + torch.mul((1 - new_label), feature_2)
        return out


class D_branch(nn.Module):
    """G input channel with instance normalization."""
    def __init__(self):
        super(D_branch, self).__init__()

        # 32*32*256 to 16*16*512
        conv_4 = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                    ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                    nn.ReflectionPad2d(1),
                    nn.AvgPool2d(kernel_size=3, stride=2),
                    nn.LeakyReLU(0.2)]

        self.conv_4 = nn.Sequential(*conv_4)

        # 16*16*512 to 8*8*1024
        self.conv_5 = Conv2dBlock(512, 1024, 4, 2, 1, norm='none', activation='lrelu', activation_first=False)

    def forward(self, x):
        current = self.conv_4(x)
        current = self.conv_5(current)
        return current


class GPPatchMcResDis_Multi_Switch_FM_up(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis_Multi_Switch_FM_up, self).__init__()

        self.c_dim = hp['num_classes']

        #128*128*3 to 128*128*64
        self.conv_1 = Conv2dBlock(3, 64, 7, 1, 3, pad_type='reflect', norm='none', activation='none')

        #128*128*64 to 64*64*128
        conv_2 = [ActFirstResBlock(64, 64, None, 'lrelu', 'none'),
                  ActFirstResBlock(64, 128, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        #64*64*128 to 32*32*256
        conv_3 = [ActFirstResBlock(128, 128, None, 'lrelu', 'none'),
                  ActFirstResBlock(128, 256, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        self.conv_2 = nn.Sequential(*conv_2)
        self.conv_3 = nn.Sequential(*conv_3)

        self.conv_domains = nn.ModuleList()
        self.conv_outs = nn.ModuleList()
        for i in range(self.c_dim):
            self.conv_domains.append(D_branch())
            self.conv_outs.append(Conv2dBlock(1024, 1, 3, 1, 1, norm='none', activation='none', use_bias=False, activation_first=False))

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))

        #label2onehot
        def label2onehot(labels, dim):
            """Convert label indices to one-hot vectors."""
            batch_size = labels.size(0)
            out = torch.zeros(batch_size, dim)
            out[np.arange(batch_size), labels.long()] = 1
            return out

        y = label2onehot(y, self.c_dim).cuda()

        features = []
        current = self.conv_1(x)
        current = self.conv_2(current)

        self.use_FM3 = True
        self.use_cFM = True

        if self.use_FM3:
            features.append(current)

        current = self.conv_3(current)

        if self.use_FM3:
            features.append(current)

        out_size = int(current.size(-1) / 4)

        current = self.feature_switching(current, y, self.conv_domains, 1024, out_size)
        if self.use_cFM:
            features.append(current)

        out = self.feature_switching(current, y, self.conv_outs, 1, out_size)

        return out, features

    def feature_switching(self, cur_feature, label, branches, out_channel, out_size):
        conv = []
        batch_size = label.size(0)
        class_num = label.size(1)

        # real or fake
        for i in range(0, class_num):
            conv.append(branches[i](cur_feature).unsqueeze(0).transpose(1, 0))  # add domain

        conv_total = torch.cat(conv, 1)  # concat and exchange domain:[batch_size,5,channel,img_size,img_size]
        conv_total = torch.mul(label.unsqueeze(2), conv_total.view(batch_size, class_num, -1))  # [batch_size,class,feature_map]
        conv_total = conv_total.view(batch_size, class_num, out_channel, out_size, out_size)  # [batch_size,class,channel,img_size,img_size]
        conv_total = conv_total.transpose(1, 0)  # [class,batch_size,channel,img_size,img_size]
        conv_picked = sum(conv_total, 0)
        conv_picked = conv_picked.view(batch_size, out_channel, out_size, out_size)
        return conv_picked


class GPPatchMcResDis_Switch_FM_up_class(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis_Switch_FM_up_class, self).__init__()

        #128*128*3 to 128*128*64
        self.conv_1 = Conv2dBlock(3, 64, 7, 1, 3, pad_type='reflect', norm='none', activation='none')

        #128*128*64 to 64*64*128
        conv_2 = [ActFirstResBlock(64, 64, None, 'lrelu', 'none'),
                  ActFirstResBlock(64, 128, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        #64*64*128 to 32*32*256
        conv_3 = [ActFirstResBlock(128, 128, None, 'lrelu', 'none'),
                  ActFirstResBlock(128, 256, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2)]

        #32*32*256 to 16*16*512
        conv_4_A = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                  ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                  nn.ReflectionPad2d(1),
                  nn.AvgPool2d(kernel_size=3, stride=2),
                  nn.LeakyReLU(0.2)]

        conv_4_B = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                    ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                    nn.ReflectionPad2d(1),
                    nn.AvgPool2d(kernel_size=3, stride=2),
                    nn.LeakyReLU(0.2)]

        #16*16*512 to 8*8*1024
        self.conv_A = Conv2dBlock(512, 1024, 4, 2, 1, norm='none', activation='lrelu', activation_first=False)
        self.conv_A_out = Conv2dBlock(1024, 1, 3, 1, 1, norm='none', activation='none', use_bias=False, activation_first=False)

        #16*16*512 to 8*8*1024
        self.conv_B = Conv2dBlock(512, 1024, 4, 2, 1, norm='none', activation='lrelu', activation_first=False)
        self.conv_B_out = Conv2dBlock(1024, 1, 3, 1, 1, norm='none', activation='none', use_bias=False, activation_first=False)

        self.conv_2 = nn.Sequential(*conv_2)
        self.conv_3 = nn.Sequential(*conv_3)
        self.conv_4_A = nn.Sequential(*conv_4_A)
        self.conv_4_B = nn.Sequential(*conv_4_B)

        #32*32*256 to 16*16*512
        conv_class = [ActFirstResBlock(256, 256, None, 'lrelu', 'none'),
                            ActFirstResBlock(256, 512, None, 'lrelu', 'none'),
                            nn.ReflectionPad2d(1),
                            nn.AvgPool2d(kernel_size=3, stride=2),
                            nn.LeakyReLU(0.2),

                           # 16*16*512 to 8*8*1024
                           ActFirstResBlock(512, 512, None, 'lrelu', 'none'),
                           ActFirstResBlock(512, 512, None, 'lrelu', 'none'),
                           nn.ReflectionPad2d(1),
                           nn.AvgPool2d(kernel_size=3, stride=2),
                           nn.LeakyReLU(0.2),

                           # 8*8*512 to 4*4*1024
                           ActFirstResBlock(512, 512, None, 'lrelu', 'none'),
                           ActFirstResBlock(512, 1024, None, 'lrelu', 'none'),
                           nn.ReflectionPad2d(1),
                           nn.AvgPool2d(kernel_size=3, stride=2),
                           nn.LeakyReLU(0.2),

                           nn.Conv2d(1024, 2, kernel_size=4, stride=1, padding=0, bias=False)]

        self.conv_class = nn.Sequential(*conv_class)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        features = []
        current = self.conv_1(x)
        current = self.conv_2(current)

        self.use_FM3 = True
        self.use_cFM = True

        if self.use_FM3:
            features.append(current)

        current = self.conv_3(current)

        if self.use_FM3:
            features.append(current)

        feat_A = self.conv_4_A(current)
        feat_B = self.conv_4_B(current)

        out_A = self.conv_A(feat_A)
        out_B = self.conv_B(feat_B)

        if self.use_cFM:
            cFeat = self.feature_switch(out_A, out_B, y)
            features.append(cFeat)

        out_A = self.conv_A_out(out_A)
        out_B = self.conv_B_out(out_B)

        out = self.feature_switch(out_A, out_B, y)

        out_cls = self.conv_class(current)

        return out, features, out_cls.view(out_cls.size(0), out_cls.size(1))

    def feature_switch(self, feature_1, feature_2, label):
        new_label = []
        for y_binary in label:
            if int(y_binary) == 1:
                new_label.append(torch.ones(feature_1.size()[1:]).unsqueeze(0))
            else:
                new_label.append(torch.zeros(feature_1.size()[1:]).unsqueeze(0))

        new_label = torch.cat(new_label, 0).cuda()
        out = torch.mul(new_label, feature_1) + torch.mul((1 - new_label), feature_2)
        return out


class FewShotGen(nn.Module):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']

        self.Switch_1 = False
        self.Switch_2 = False

        self.ada_up_num = 3

        self.ada_para_num = 2*2*512
        if self.ada_up_num>0:
            self.ada_para_num += 2*256
        if self.ada_up_num>1:
            self.ada_para_num += 2*128
        if self.ada_up_num>2:
            self.ada_para_num += 2*64

        if self.Switch_1:
            self.enc_class_model_A = SA_style_Encoder(down_class, 3, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')
            self.enc_class_model_B = SA_style_Encoder(down_class, 3, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')
            self.enc_class_model_com = SA_style_Encoder(down_class, 3, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')
        else:
            self.enc_class_model_com = SA_style_Encoder(down_class, 3, nf, latent_dim, norm='none', activ='relu', pad_type='reflect')

        if self.Switch_2:
            self.mlp_A = MLP(latent_dim, self.ada_para_num, nf_mlp, n_mlp_blks, norm='none', activ='relu')
            self.mlp_B = MLP(latent_dim, self.ada_para_num, nf_mlp, n_mlp_blks, norm='none', activ='relu')
            self.mlp_com = MLP(latent_dim, self.ada_para_num, nf_mlp, n_mlp_blks, norm='none', activ='relu')
        else:
            self.mlp_com = MLP(latent_dim, self.ada_para_num, nf_mlp, n_mlp_blks, norm='none', activ='relu')

        self.use_attn_encoder = False
        self.use_attn_encoder_single = False

        if self.use_attn_encoder:
            self.enc_content = Atten_Encoder(down_content,
                                              n_res_blks,
                                              3,
                                              nf,
                                              'in',
                                              activ='relu',
                                              pad_type='reflect')

        elif self.use_attn_encoder_single:
            self.enc_content = Atten_Encoder_single(down_content,
                                              n_res_blks,
                                              3,
                                              nf,
                                              'in',
                                              activ='relu',
                                              pad_type='reflect')
        else:
            self.enc_content = ContentEncoder(down_content,
                                              n_res_blks,
                                              3,
                                              nf,
                                              'in',
                                              activ='relu',
                                              pad_type='reflect')

        self.dec = Decoder_up(down_content,
                           n_res_blks,
                           self.enc_content.output_dim,
                           3,
                           res_norm='adain',
                           activ='relu',
                           pad_type='reflect')

    def forward(self, one_image, model_set):
        # reconstruct an image
        content, model_codes = self.encode(one_image, model_set)
        model_code = torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code)
        return images_trans

    def encode(self, one_image, model_set):
        # extract content code from the input image
        content = self.enc_content(one_image)
        # extract model code from the images in the model set
        class_codes = self.enc_class_model(model_set)
        class_code = torch.mean(class_codes, dim=0).unsqueeze(0)
        return content, class_code

    def decode(self, content, style_code, style):
        images = self.dec(content, style_code, style)
        return images

    def Switch_encode(self, model_code, label):
        if self.Switch_1:
            class_codes_A, class_feature_A  = self.enc_class_model_A(model_code)
            class_codes_B, class_feature_B = self.enc_class_model_B(model_code)
            class_codes_com, class_feature_com = self.enc_class_model_com(model_code)
            class_codes = self.feature_switch(class_codes_A, class_codes_B, label) + class_codes_com
            class_feature = self.feature_switch(class_feature_A, class_feature_B, label) + class_feature_com
        else:
            class_codes, class_feature = self.enc_class_model_com(model_code)

        if self.Switch_2:
            para_code_A = self.mlp_A(class_codes)
            para_code_B = self.mlp_B(class_codes)
            para_code_com = self.mlp_com(class_codes)
            para_code = self.feature_switch(para_code_A, para_code_B, label) + para_code_com
        else:
            para_code = self.mlp_com(class_codes)
        return para_code, class_feature

    def feature_switch(self, feature_1, feature_2, label):
        new_label = []
        for y_binary in label:
            if int(y_binary) == 1:
                new_label.append(torch.ones(feature_1.size()[1:]).unsqueeze(0))
            else:
                new_label.append(torch.zeros(feature_1.size()[1:]).unsqueeze(0))

        new_label = torch.cat(new_label, 0).cuda()
        out = torch.mul(new_label, feature_1) + torch.mul((1 - new_label), feature_2)
        return out


class SA_style_Encoder(nn.Module):
    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(SA_style_Encoder, self).__init__()
        self.model = []
        #128*128*3 to 128*128*64
        self.model += [Conv2dBlock(ind_im, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        #128*128*64 to 64*64*128 to 32*32*256 to 16*16*512
        for i in range(3):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2

        self.para_model = [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
                            nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(dim, latent_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.para_model = nn.Sequential(*self.para_model)
        self.output_dim = dim

    def forward(self, x):
        style_feature = self.model(x)
        stlye_codes = self.para_model(style_feature)
        return stlye_codes, style_feature


class Content_classifier(nn.Module):
    def __init__(self):
        super(Content_classifier, self).__init__()    #input; [B, 512, 16, 16]
        cnn_f = [nn.ReflectionPad2d(1),
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=0),
                 nn.LeakyReLU(0.2),
                 nn.Conv2d(1024, 1, kernel_size=8, stride=1, padding=0, bias=False)]
        self.cnn_f = nn.Sequential(*cnn_f)

    def forward(self, x):
        out = self.cnn_f(x)
        return out.view(out.size(0), 1)


class ContentEncoder(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Atten_Encoder(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(Atten_Encoder, self).__init__()
        self.model_1 = []
        self.attn_1_1 = []
        self.attn_1_2 = []
        self.attn_2_1 = []
        self.attn_2_2 = []
        self.attn_3_1 = []
        self.attn_3_2 = []

        # 128*128*3 to 64*64*128
        self.model_1 += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type),
                         Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.attn_1_1 = [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type),
                        Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.attn_1_2 = [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type),
                          Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.attn_1_com = [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type),
                         Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        dim = dim * 2

        # 64*64*128 to 32*32*256
        self.model_2 = Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_2_1 = Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_2_2 = Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_2_com = Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)

        #32*32*256 to 16*16*512
        self.model_3 = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_3_1 = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_3_2 = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_3_com = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)

        dim = 4 * dim

        # resblock
        self.model_4 = ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)

        self.model_1 = nn.Sequential(*self.model_1)
        self.attn_1_1 = nn.Sequential(*self.attn_1_1)
        self.attn_1_2 = nn.Sequential(*self.attn_1_2)
        self.attn_1_com = nn.Sequential(*self.attn_1_com)

        self.output_dim = dim

    def forward(self, x, content_label):
        current = self.model_1(x)

        current_attn_1 = self.attn_1_1(x)
        current_attn_2 = self.attn_1_2(x)
        current_attn_com = self.attn_1_com(x)
        current_attn = 0.5 * (self.feature_switch(current_attn_1, current_attn_2, content_label) + current_attn_com)

        current = current * current_attn

        current = self.model_2(current)
        current_attn_1 = self.attn_2_1(current_attn_1)
        current_attn_2 = self.attn_2_2(current_attn_2)
        current_attn_com = self.attn_2_com(current_attn_com)
        current_attn = 0.5 * (self.feature_switch(current_attn_1, current_attn_2, content_label) + current_attn_com)

        current = current * current_attn

        current = self.model_3(current)
        current_attn_1 = self.attn_3_1(current_attn_1)
        current_attn_2 = self.attn_3_2(current_attn_2)
        current_attn_com = self.attn_3_com(current_attn_com)
        current_attn = 0.5 * (self.feature_switch(current_attn_1, current_attn_2, content_label) + current_attn_com)

        current = current * current_attn

        out = self.model_4(current)
        return out

    def feature_switch(self, feature_1, feature_2, label):
        new_label = []
        for y_binary in label:
            if int(y_binary) == 1:
                new_label.append(torch.ones(feature_1.size()[1:]).unsqueeze(0))
            else:
                new_label.append(torch.zeros(feature_1.size()[1:]).unsqueeze(0))

        new_label = torch.cat(new_label, 0).cuda()
        out = torch.mul(new_label, feature_1) + torch.mul((1 - new_label), feature_2)
        return out


class Atten_Encoder_single(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(Atten_Encoder_single, self).__init__()
        self.model_1 = []
        self.attn_1 = []
        self.attn_2 = []
        self.attn_3 = []

        # 128*128*3 to 64*64*128
        self.model_1 += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type),
                         Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.attn_1 = [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type),
                          Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        dim = dim * 2

        # 64*64*128 to 32*32*256
        self.model_2 = Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_2 = Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)

        #32*32*256 to 16*16*512
        self.model_3 = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.attn_3 = Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        dim = 4 * dim

        # resblock
        self.model_4 = ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)

        self.model_1 = nn.Sequential(*self.model_1)
        self.attn_1 = nn.Sequential(*self.attn_1)

        self.output_dim = dim

    def forward(self, x):
        current = self.model_1(x)
        current_attn = self.attn_1(x)
        current = current * current_attn

        current = self.model_2(current)
        current_attn = self.attn_2(current_attn)
        current = current * current_attn

        current = self.model_3(current)
        current_attn = self.attn_3(current_attn)
        current = current * current_attn

        out = self.model_4(current)
        return out


class Decoder_up(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder_up, self).__init__()
        self.n_res = n_res
        self.model = []
        self.model_1 = []
        self.model_2 = []
        self.model_3 = []

        self.use_ada_up_1 = False
        self.use_ada_up_2 = True
        self.ada_up_num = 2

        self.res = False

        self.norm_type = 'siln'

        for i in range(n_res):
            if self.norm_type is 'siln':
                if not self.res:
                    setattr(self, 'AdaSILNBlock_' + str(i + 1), AdaSILNBlock(dim, use_bias=False))
                else:
                    setattr(self, 'AdaSILNBlock_' + str(i + 1), ResnetAdaSILNBlock(dim, use_bias=False))

            elif self.norm_type is 'iln':
                if not self.res:
                    setattr(self, 'AdaSILNBlock_' + str(i + 1), AdaILNBlock(dim, use_bias=False))
                else:
                    setattr(self, 'AdaSILNBlock_' + str(i + 1), ResnetAdaILNBlock(dim, use_bias=False))
            else:
                if not self.res:
                    setattr(self, 'AdaSILNBlock_' + str(i + 1), AdaINBlock(dim, use_bias=False))
                else:
                    setattr(self, 'AdaSILNBlock_' + str(i + 1), ResnetAdaINBlock(dim, use_bias=False))

        self.norm_type = 'siln'

        for i in range(ups):
            if (self.use_ada_up_1 or self.use_ada_up_2) and (i==0) and self.ada_up_num>0:
                self.model_1 += [nn.Upsample(scale_factor=2)]
                if self.use_ada_up_2:
                    self.model_1 += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=self.norm_type, activation=activ, pad_type=pad_type),
                                     Conv2dBlock(dim//2, dim // 2, 5, 1, 2, norm='none', activation='none', pad_type=pad_type)]
                else:
                    self.model_1 += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='none', activation='none', pad_type=pad_type)]
                if self.norm_type is 'siln':
                    self.adaptive_norm_1 = adaSILN(dim//2)
                elif self.norm_type is 'iln':
                    self.adaptive_norm_1 = adaILN(dim//2)
                else:
                    self.adaptive_norm_1 = adaIN(dim//2)
                self.relu_1 = nn.ReLU(inplace=True)
                self.model_1 = nn.Sequential(*self.model_1)

            elif (self.use_ada_up_1 or self.use_ada_up_2) and (i==1) and self.ada_up_num>1:
                self.model_2 += [nn.Upsample(scale_factor=2)]
                if self.use_ada_up_2:
                    self.model_2 += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=self.norm_type, activation=activ, pad_type=pad_type),
                                     Conv2dBlock(dim//2, dim // 2, 5, 1, 2, norm='none', activation='none', pad_type=pad_type)]
                else:
                    self.model_2 += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='none', activation='none', pad_type=pad_type)]
                if self.norm_type is 'siln':
                    self.adaptive_norm_2 = adaSILN(dim // 2)
                elif self.norm_type is 'iln':
                    self.adaptive_norm_2 = adaILN(dim // 2)
                else:
                    self.adaptive_norm_2 = adaIN(dim // 2)
                self.relu_2 = nn.ReLU(inplace=True)
                self.model_2 = nn.Sequential(*self.model_2)

            elif (self.use_ada_up_1 or self.use_ada_up_2) and (i==2) and self.ada_up_num>2:
                self.model_3 += [nn.Upsample(scale_factor=2)]
                if self.use_ada_up_2:
                    self.model_3 += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=self.norm_type, activation=activ, pad_type=pad_type),
                                     Conv2dBlock(dim//2, dim // 2, 5, 1, 2, norm='none', activation='none', pad_type=pad_type)]
                else:
                    self.model_3 += [Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='none', activation='none', pad_type=pad_type)]
                if self.norm_type is 'siln':
                    self.adaptive_norm_3 = adaSILN(dim // 2)
                elif self.norm_type is 'iln':
                    self.adaptive_norm_3 = adaILN(dim // 2)
                else:
                    self.adaptive_norm_3 = adaIN(dim // 2)
                self.relu_3 = nn.ReLU(inplace=True)
                self.model_3 = nn.Sequential(*self.model_3)
            else:
                self.model += [nn.Upsample(scale_factor=2),
                               Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=self.norm_type, activation=activ, pad_type=pad_type)]

            dim //= 2

        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, ada_para, style):
        for i in range(self.n_res):
            x = getattr(self, 'AdaSILNBlock_' + str(i+1))(x, style, ada_para[:, i*1024: i*1024+512], ada_para[:, i*1024+512: (i+1)*1024])

        if (self.use_ada_up_1 or self.use_ada_up_2):
            if self.ada_up_num>0:
                x = self.model_1(x)
                x = self.adaptive_norm_1(x, ada_para[:, 2048: 2048+256*1], ada_para[:, 2048+256*1: 2048+256*2])
                x = self.relu_1(x)
            if self.ada_up_num > 1:
                x = self.model_2(x)
                x = self.adaptive_norm_2(x, ada_para[:, 2048+256*2: 2048+256*2+128*1], ada_para[:, 2048+256*2+128*1:2048+256*2+128*2])
                x = self.relu_2(x)
            if self.ada_up_num > 2:
                x = self.model_3(x)
                x = self.adaptive_norm_3(x, ada_para[:, 2048+256*2+128*2: 2048+256*2+128*2+64*1], ada_para[:, 2048+256*2+128*2+64*1: 2048+256*2+128*2+64*2])
                x = self.relu_3(x)

        return self.model(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))