# -*- codeing = utf -8 -*-
# @Time : 2022/7/24 22:37
# @Author : 云
# @File : __init__.py.py
# @Software : PyCharm
import torch
from .CHA_Net import CHA_Net
import argparse
def modify_commandline_options(parser, is_train=True):
    parser.add_argument('--stn_cfg', type=str, default='A', help='Set the configuration used to build the STN.')
    parser.add_argument('--stn_type', type=str, default='unet',
                        help='The type of STN to use. Currently supported are [unet, affine]')
    return parser


def define_stn(opt, stn_type='DIR'):
    """Create and return an STN model with the relevant configuration."""
    def wrap_multigpu(stn_module, opt):
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            stn_module.to(opt.gpu_ids[0])
            stn_module = torch.nn.DataParallel(stn_module, opt.gpu_ids)  # multi-GPUs
        return stn_module

    nc_a = opt.input_nc if opt.direction == 'AtoB' else opt.output_nc
    nc_b = opt.output_nc if opt.direction == 'AtoB' else opt.input_nc
    height = opt.img_height
    width = opt.img_width
    cfg = opt.stn_cfg
    stn = None
    if stn_type == 'DIR':
        net_R = CHA_Net()

    return wrap_multigpu(net_R, opt)
