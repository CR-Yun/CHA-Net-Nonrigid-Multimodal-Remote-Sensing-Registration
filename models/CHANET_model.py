import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from util.tb_visualizer import TensorboardVisualizer
from . import networks
from .base_model import BaseModel
import models.network as network
from .loss.reg_losses import LossFunction_Dense
from .network import losses as losses
from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform



class CHANETModel(BaseModel):
    """
        This is the official implementation of the CHA-Net model,
        used for non-rigid multimodal image registration,
        referencing the implementation of pix2pix:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Modify the command line."""
        parser = network.modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Weight for the GAN loss.')
            parser.add_argument('--lambda_recon', type=float, default=100.0,
                                help='Weight for the L1 reconstruction loss.')
            parser.add_argument('--lambda_smooth', type=float, default=210, help='Regularization term used by the STN')#原程序是0.0但是论文里写的是200 这个是平滑性损失 可以减少过度的形变改成230了
            parser.add_argument('--enable_tbvis', action='store_true',
                                help='Enable tensorboard visualizer (default : False)')
            parser.add_argument('--multi_resolution', type=int, default=1,
                                help='Use of multi-resolution discriminator.'
                                     '(if equals to 1 then no multi-resolution training is applied)')
            TensorboardVisualizer.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.criterion = LossFunction_Dense().to(device=self.device)
        self.train_stn = True
        self.setup_visualizers()
        if self.isTrain and opt.enable_tbvis:
            self.tb_visualizer = TensorboardVisualizer(self, ['netR'], self.loss_names, self.opt)
        else:
            self.tb_visualizer = None
        self.define_networks()
        if self.tb_visualizer is not None:
            print('Enabling Tensorboard Visualizer!')
            self.tb_visualizer.enable()
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSSIM = losses.SSIM_loss(False)
            self.criterionGrad = losses.Grad('l2')
            self.setup_optimizers()
            self.FL = FocalLoss(class_num=2, gamma=5)
            self.affine = AffineTransform(translate=0.01).to(self.device)
            self.elastic = ElasticTransform(kernel_size=101, sigma=18).to(self.device)
            self.criterionMI = losses.MutualInformation()

    def setup_visualizers(self):
        self.loss_names = ['R']
        self.visual_names = ['result']
        model_names = ['R','D']
        self.model_names = model_names

    def define_networks(self):

        opt = self.opt
        AtoB = opt.direction == 'AtoB'
        self.netR = network.define_stn(self.opt, 'DIR')
        self.netD = netD().to(device=self.device)
        self.netD.to(opt.gpu_ids[0])
        self.netD = torch.nn.DataParallel(self.netD, opt.gpu_ids)


    def setup_optimizers(self):
        opt = self.opt
        self.optimizer_R = torch.optim.Adam(itertools.chain(self.netR.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999), )
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.optimizers.append(self.optimizer_R)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        if AtoB:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']
        else:
            self.real_A = input['B'].to(self.device)
            self.real_B = input['A'].to(self.device)
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def _warp_Dense_loss_unsupervised(criterion, im_pre, im_fwarp, im_fix, im_warp, flow):
        total_loss, multi, ncc, grad = criterion(im_pre, im_fwarp, im_fix, im_warp, flow)

        return multi, ncc, grad, total_loss


    def backward_R(self,target):

        ir_affine, affine_disp = self.affine(self.real_B)

        ir_elastic, elastic_disp = self.elastic(ir_affine)

        self.real_B = ir_elastic.detach()


        B2A_warp, A2B_warp, flow, int_flow1, int_flow2, disp_pre,style_b,style_a,content_b,content_a,loss_ha=self.netR(self.real_A,self.real_B)

        if target:

            style_b_d=self.netD(style_b)
            style_a_d=self.netD(style_a)

            model_a = Variable(torch.ones(style_b_d.size(0)).long().cuda())
            loss_a_d = 0.5 * self.FL(style_b_d, model_a)
            model_b = Variable(torch.zeros(style_a.size(0)).long().cuda())
            loss_b_d = 0.5 * self.FL(style_a_d, model_b)
            loss_D = loss_a_d + loss_b_d
            Mutual_content = F.avg_pool2d(content_b, (8, 8)).view(content_b.size(0), -1)
            Mutual_style = F.avg_pool2d(style_b, (8, 8)).view(style_b.size(0), -1)
            Mutual_content = F.normalize(Mutual_content,dim=1)
            Mutual_style = F.normalize(Mutual_style,dim=1)
            Mutual_loss = -self.criterionMI(Mutual_content, Mutual_style)

        else:

            style_b_d=self.netD(style_b)
            style_a_d=self.netD(style_a)

            model_a = Variable(torch.zeros(style_b_d.size(0)).long().cuda())
            loss_a_d = 0.5 * self.FL(style_b_d, model_a)
            model_b = Variable(torch.zeros(style_a.size(0)).long().cuda())
            loss_b_d = 0.5 * self.FL(style_a_d, model_b)
            loss_D = loss_a_d + loss_b_d

            Mutual_content = F.avg_pool2d(content_b, (8, 8)).view(content_b.size(0), -1)
            Mutual_style = F.avg_pool2d(style_b, (8, 8)).view(style_b.size(0), -1)
            Mutual_content = F.normalize(Mutual_content, dim=1)
            Mutual_style = F.normalize(Mutual_style, dim=1)
            Mutual_loss = -self.criterionMI(Mutual_content, Mutual_style)

        self.result = B2A_warp


        total_loss, multi, ncc, grad = self.criterion(B2A_warp,A2B_warp,self.real_A,self.real_B,flow)
        loss1, loss2, grad_loss, loss= multi, ncc, grad, total_loss
        lambdaha = 0.15
        loss = loss + lambdaha * loss_ha
        self.loss_R = loss
        return loss,loss_D,Mutual_loss

    def grad_reverse(self,x):
        return GradReverse.apply(x)

    def optimize_parameters(self, target):

        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        self.forward()
        self.set_requires_grad(self.netR,True)
        self.set_requires_grad(self.netD,True)
        self.optimizer_R.zero_grad()
        self.optimizer_D.zero_grad()

        loss,loss_D,loss_MI = self.backward_R(target)
        loss = loss + loss_D + loss_MI
        loss.backward()
        self.optimizer_R.step()
        self.optimizer_D.step()
        if self.tb_visualizer is not None:
            self.tb_visualizer.iteration_step()

    def test_R(self):

        B2A_warp, A2B_warp, flow, int_flow1, int_flow2, disp_pre,style_b,style_a,content_b,content_a,loss_ha=self.netR(self.real_A,self.real_B)
        self.result = B2A_warp



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * (-1)

class netD(nn.Module):
    def __init__(self):#,context=False
        super(netD, self).__init__()
        #输入是 torch.Size([16, 32, 64, 64])
        self.conv1 = nn.Conv2d(32, 24, kernel_size=3, stride=2,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3, stride=2,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3, stride=2,padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(6)
        self.fc = nn.Linear(6,2)
        # self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,6)
        x = self.fc(x)
        return x

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(
        self,
        class_num,
        alpha=None,
        gamma=2,
        size_average=True,
        sigmoid=False,
        reduce=True,
    ):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            # F.softmax(inputs)
            if targets == 0:
                probs = 1 - P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        else:
            # inputs = F.sigmoid(inputs)
            P = F.softmax(inputs,dim=1)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.0)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
