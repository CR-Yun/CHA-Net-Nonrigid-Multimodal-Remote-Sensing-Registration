"""
The code implementation of CHA-Net is based on pix2pix and cyclegan.
See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import time
import torch
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.visualizer import save_images
import os
from util import html


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    opt.dataset_mode = 'saligned'
    dataset_s = create_dataset(opt) # enhanced dataset
    dataset_s_size = len(dataset_s)

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format("test", opt.epoch))  # define the website directory

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count,opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        model.train()
        dataset_s_enumerate = list(enumerate(dataset_s))
        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(target = True)  # calculate loss functions, get gradients, update network weights

            _, data_s = dataset_s_enumerate[i] # Monomodal dataset augmentation for registration robustness
            model.set_input(data_s)
            model.optimize_parameters(target = False)

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file

                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        model.update_learning_rate()    # update learning rates in the ending of every epoch.


        model.save_networks('latest')
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if model.tb_visualizer:  # Notify tensorboard visualizer on each end of epoch.
            model.tb_visualizer.epoch_step()

    if model.tb_visualizer:
        model.tb_visualizer.end()
