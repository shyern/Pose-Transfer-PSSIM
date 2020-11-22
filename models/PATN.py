import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
from losses.PerceptualLoss import PerceptualLoss
from losses.other_losses.PerceptualSSIMLoss import PerceptualSSIMLoss
from losses.other_losses.StyleLoss import StyleLoss
from losses.L1_plus_perceptual_styleLoss import L1_plus_perceptual_styleLoss

from losses.pytorch_msssim import SSIM, MS_SSIM, FPart_BSSIM


class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_BP2_mask_set = self.Tensor(nb, opt.BP_limbs_nc, size, size)
        # self.input_BP2_KC_set = self.Tensor(nb, opt.BP_input_nc, 2)

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_blocks=opt.n_blocks, n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.L1_type == 'origin':  # L1 loss
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'perL1':  # PerL1 loss
                self.criterionL1 = PerceptualLoss(opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            elif opt.L1_type == 'SSIM':  # SSIM loss
                self.criterionSSIM = SSIM(win_size=opt.win_size,win_sigma=opt.win_sigma,data_range=1.0, size_average=True)
            elif opt.L1_type == 'Style':  # Style loss
                self.criterionStyle = StyleLoss(opt.lambda_style, opt.perceptual_layers, self.gpu_ids)
            elif opt.L1_type == 'PerSSIM':  # PerceptualSSIM loss
                self.criterionPerSSIM = PerceptualSSIMLoss(opt.lambda_perssim, opt.perceptual_layers, self.gpu_ids, win_size=opt.win_size, win_sigma=opt.win_sigma)
            elif opt.L1_type == 'l1_plus_perL1':  # L1 + PerL1 loss
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)

            elif opt.L1_type == 'FPart_BSSIM_plus_perL1_L1':  # FPart_BSSIM + PerL1 + L1 loss
                self.criterionSSIM = FPart_BSSIM(data_range=1.0, size_average=True, win_size=opt.win_size, win_sigma=opt.win_sigma)
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids,
                                                      opt.percep_is_l1)
            elif opt.L1_type == 'FPart_BSSIM_plus_perL1_style':  #FPart_BSSIM + PerL1 + style loss
                self.criterionSSIM = FPart_BSSIM(data_range=1.0, size_average=True, win_size=opt.win_size, win_sigma=opt.win_sigma)
                self.criterionL1 = L1_plus_perceptual_styleLoss(opt.lambda_style, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
                # self.criterionL1 = torch.nn.L1Loss()
            else:
                self.criterionL1 = None
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # if self.isTrain:
        #     if opt.with_D_PB:
        #         networks.print_network(self.netD_PB)
        #     if opt.with_D_PP:
        #         networks.print_network(self.netD_PP)
        # print('-----------------------------------------------')

    def set_input(self, input):
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']
        # input_BP2_KC = input['BP2_KC']
        input_BP2_mask = input['BP2_mask']

        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)
        self.input_BP2_mask_set.resize_(input_BP2_mask.size()).copy_(input_BP2_mask)
        # self.input_BP2_KC_set.resize_(input_BP2_KC.size()).copy_(input_BP2_KC)
        # self.input_BP2_KC_set = input_BP2_KC

        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]


    def forward(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.fake_p2 = self.netG(G_input)


    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.fake_p2 = self.netG(G_input)


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_G(self):
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        # L1 loss
        if self.opt.L1_type == 'origin':
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A  # l1 loss
        elif self.opt.L1_type == 'perL1':
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2)  # perL1 loss
        elif self.opt.L1_type == 'SSIM':
            self.loss_G_L1 = (1-self.criterionSSIM(self.fake_p2, self.input_P2)) * self.opt.lambda_SSIM  # ssim loss
        elif self.opt.L1_type == 'Style':
            self.loss_G_L1 = self.criterionStyle(self.fake_p2, self.input_P2)  #  style loss
        elif self.opt.L1_type == 'PerSSIM':
            self.loss_G_L1 = self.criterionPerSSIM(self.fake_p2, self.input_P2)  # Perceptual SSIM loss
        elif self.opt.L1_type == 'l1_plus_perL1':
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            self.loss_originL1 = losses[1].item()
            self.loss_perceptual = losses[2].item()
        elif self.opt.L1_type == 'FPart_BSSIM_plus_perL1_L1':
            self.loss_ssim = (1 - self.criterionSSIM(self.fake_p2,
                                                     self.input_P2, self.input_BP2_mask_set)) * self.opt.lambda_SSIM
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]+self.loss_ssim   #  perL1 + L1 loss + fpart_bssim loss
            self.loss_originL1 = losses[1].item()  # L1 loss
            self.loss_perceptual = losses[2].item()  # perL1 loss
        elif self.opt.L1_type == 'FPart_BSSIM_plus_perL1_style':
            self.loss_ssim = (1 - self.criterionSSIM(self.fake_p2,
                                                     self.input_P2, self.input_BP2_mask_set)) * self.opt.lambda_SSIM
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]+self.loss_ssim  # perL1 + style loss + fpart_bssim loss
            self.loss_style = losses[1].item()  # style loss
            self.loss_perceptual = losses[2].item()  # perL1 loss
        else:
            self.loss_G_L1 = torch.tensor(0).cuda()


        pair_L1loss = self.loss_G_L1
        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN

        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        pair_loss.backward()

        self.pair_L1loss = pair_L1loss.item()
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.item()


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
        self.loss_D_PB = loss_D_PB.item()

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)
        self.loss_D_PP = loss_D_PP.item()


    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_L1loss', self.pair_L1loss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual
        if self.opt.L1_type == 'FPart_BSSIM_plus_perL1_L1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual
            ret_errors['ssim'] = self.loss_ssim.item()
        if self.opt.L1_type == 'FPart_BSSIM_plus_perL1_style':
            ret_errors['style'] = self.loss_style
            ret_errors['perceptual'] = self.loss_perceptual
            ret_errors['ssim'] = self.loss_ssim.item()

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        fake_p2 = util.tensor2im(self.fake_p2.data)

        vis = np.zeros((height, width*5, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_P2
        vis[:, width*3:width*4, :] = input_BP2
        vis[:, width*4:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

