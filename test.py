import torch
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.visualizer import save_images
import os
from util import html


if __name__ == '__main__':
    test_path = "../data_nirandvis/combine_test"
    result_path = "./result"
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.dataroot = test_path
    test_dataset = create_dataset(opt)
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format("test", opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    with torch.no_grad():
        test_numbers = len(test_dataset)
        print("The number of test image:", test_numbers)
        for i, data in enumerate(test_dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                        width=opt.display_winsize)

