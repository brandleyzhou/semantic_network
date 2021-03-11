from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)

class SegOptions:
    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description = 'Semantic Segmentation Options')

        # model and dataset
        self.parser.add_argument('--model', type=str, default="ENet", choices = ['MONO','FPENet', 'SQNet', 'ENet'], help="model name: (default ENet)")
        self.parser.add_argument('--feature_fusing', type=bool, default = False, help="decide whether to fuse features")
        self.parser.add_argument('--dataset', type=str, default="camvid", help="dataset: cityscapes or camvid")
        self.parser.add_argument('--input_size', type=str, default="192,640", help="input size of model")
        self.parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
        self.parser.add_argument('--classes', type=int, default=19,
                            help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument('--train_type', type=str, default="trainval",
                            help="ontrain for training on train set, ontrainval for training on train+val set")
        # training hyper params
        self.parser.add_argument('--max_epochs', type=int, default=1000,
                            help="the number of epochs: 300 for train set, 350 for train+val set")
        self.parser.add_argument('--gpu_nums', type=int, default = 1,  help="the number of GPU")
        self.parser.add_argument('--random_mirror', type=bool, default=True, help="input image random mirror")
        self.parser.add_argument('--random_scale', type=bool, default=True, help="input image resize 0.5 to 2")
        self.parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
        self.parser.add_argument('--batch_size', type=int, default=8, help="the batch size is set to 16 for 2 GPUs")
        self.parser.add_argument('--optim',type=str.lower,default='adam',choices=['sgd','adam','radam','ranger'],help="select optimizer")
        self.parser.add_argument('--lr_schedule', type=str, default='warmpoly', help='name of lr schedule: poly')
        self.parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
        self.parser.add_argument('--poly_exp', type=float, default=0.9,help='polynomial LR exponent')
        self.parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
        self.parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
        self.parser.add_argument('--use_label_smoothing', action='store_true', default=False, help="CrossEntropy2d Loss with label smoothing or not")
        self.parser.add_argument('--use_ohem', action='store_true', default=False, help='OhemCrossEntropy2d Loss for cityscapes dataset')
        self.parser.add_argument('--use_lovaszsoftmax', action='store_true', default=False, help='LovaszSoftmax Loss for cityscapes dataset')
        self.parser.add_argument('--use_focal', action='store_true', default=False,help=' FocalLoss2d for cityscapes dataset')
        # cuda setting
        self.parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
        self.parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
        # checkpoint and log
        self.parser.add_argument('--resume', type=str, default="",
                            help="use this file to load last checkpoint for continuing training")
        self.parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
        self.parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
