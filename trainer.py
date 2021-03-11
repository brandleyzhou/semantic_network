from __future__ import absolute_import, division, print_function
import os,sys
#from torch.nn import DataParallel as DDP
#from torchvision.utils import save_image
#from toch.nn.parallel import DistributedDataParallel as DDP
import json
import networks
import time
import torch
from torch import optim
import torch.nn as nn
import timeit
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight, netParams
from utils.metric.metric import get_iou
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth,\
    ProbOhemCrossEntropy2d, FocalLoss2d
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR

#from model.ENet_network import encoder
#from model.ENet_network import decoder

#from model import hr_networks
 
class Trainer:
    def __init__(self, options):
        self.opts = options
        self.lossTr_list = []
        self.mIOU_val_list = []
        self.epochs_list = []
        self.device = 'cuda'
        h, w = map(int, self.opts.input_size.split(','))
        input_size = (h, w)
        print("=====> input size:{}".format(input_size))

################################ Saving_Path #######################################

        self.opts.savedir = (self.opts.savedir + self.opts.dataset + '/' + self.opts.model + 'bs' + str(self.opts.batch_size) + 'gpu' + str(self.opts.gpu_nums) + "_" + str(self.opts.train_type) + '/')
        
        if not os.path.exists(self.opts.savedir):
            os.makedirs(self.opts.savedir)

#################################### Network #######################################

        #model = build_model(self.opts.model, num_classes=self.opts.classes)
        self.parameters_to_learn = []
        self.model = {}
        
        if self.opts.model == 'ENet':

            from model.ENet_network import encoder, decoder
            from model import hr_networks
            self.model['encoder'] = encoder.ENet_Encoder()
            self.model['decoder'] = decoder.ENet_Decoder()
            self.model['mono_encoder'] = hr_networks.ResnetEncoder(self.opts.num_layers , True)
            #encoder_path = os.path.join(model_path, "encoder_{}.pth".format(args.epoch))
            encoder_path = 'model/mono_encoder.pth'
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.model['mono_encoder'].state_dict()}
            self.model['mono_encoder'].load_state_dict(filtered_dict_enc)
            self.model['mono_encoder'].to(self.device)
            self.model['mono_encoder'].eval()
        
        elif self.opts.model == 'MONO':

            from model import hr_networks
            self.model['encoder'] = hr_networks.ResnetEncoder(self.opts.num_layers , True)
            self.model['decoder'] = hr_networks.DepthDecoder(self.model['encoder'].num_ch_enc)
        
        elif self.opts.model == 'FPENet':
            
            from model.FPENet_network import encoder,decoder
            self.model['encoder'] = encoder.FPENet_Encoder()
            
            self.model['decoder'] = decoder.FPENet_Decoder()
        else:
            print('No model')
###################################################
        self.parameters_to_learn += list(self.model['encoder'].parameters())
        self.parameters_to_learn += list(self.model['decoder'].parameters())
        
        init_weight(self.model['encoder'], nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')
        init_weight(self.model['decoder'], nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')
        
        
        ##### initialization
        #init_weight(model, nn.init.kaiming_normal_,
        #        nn.BatchNorm2d, 1e-3, 0.1,
        #        mode='fan_in')
        
        print("=====> computing network parameters and FLOPs")
        #total_paramters = netParams(model)
        #print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
        
#################################### Dataset and Loader ############################

        datas,self.trainLoader,self.valLoader = build_dataset_train(self.opts.dataset, input_size, self.opts.batch_size, self.opts.train_type, self.opts.random_scale, self.opts.random_mirror, self.opts.num_workers)

        self.opts.per_iter = len(self.trainLoader)
        self.opts.max_iter = self.opts.max_epochs * self.opts.per_iter
        print('=====> Dataset statistics')
        print("data['classWeights']: ", datas['classWeights'])
        print('mean and std:', datas['mean'], datas['std'])

##################################### Loss Function ################################

        weight = torch.from_numpy(datas['classWeights'])
        ignore_label = 255
        if self.opts.dataset == 'camvid':
            self.criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
        elif self.opts.dataset == 'camvid' and self.opts.use_label_smoothing:
            self.criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)

        elif self.opts.dataset == 'cityscapes' and self.opts.use_ohem:
            min_kept = int(self.opts.batch_size // len(self.opts.gpus) * h * w // 16)
            self.criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
        elif self.opts.dataset == 'cityscapes' and self.opts.use_label_smoothing:
            self.criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
        elif self.opts.dataset == 'cityscapes' and self.opts.use_lovaszsoftmax:
            self.criteria = LovaszSoftmax(ignore_index=ignore_label)
        elif self.opts.dataset == 'cityscapes' and self.opts.use_focal:
            self.criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)
        else:
            raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % self.opts.dataset)
        
        self.criteria = self.criteria.cuda()
        
        #self.model = model.cuda()
        
        for key,model in self.model.items():
            model.cuda()
###################################### Optiomizer ##################################

        if self.opts.optim == 'sgd':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opts.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opts.optim == 'adam':
            #optimizer = torch.optim.Adam(
            #    filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters_to_learn), lr=self.opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        elif self.opts.optim == 'radam':
            optimizer = RAdam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opts.lr, betas=(0.90, 0.999), eps=1e-08, weight_decay=1e-4)
        elif self.opts.optim == 'ranger':
            optimizer = Ranger(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opts.lr, betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4)
        elif self.opts.optim == 'adamw':
            optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.optimizer = optimizer
    
    def train(self):
        self.start_epoch = 0
        for self.epoch in range(self.opts.max_epochs):
            self.epochs_list.append(self.epoch) 
            lossTr, lr = self.run_epoch()
            self.lossTr_list.append(lossTr)
            mIOU_val, per_class_iu = self.val()
            self.mIOU_val_list.append(mIOU_val)
            print("peoch done")
            if self.epoch % 50 == 0:
                self.save_model()

        print("Training Completed")

    def run_epoch(self):
        """
        self.opts:
           train_loader: loaded for training dataset
           model: model
           criterion: loss function
           optimizer: optimization algorithm, such as ADAM or SGD
           epoch: epoch number
        return: average loss, per class IoU, and mean IoU
        """
        print("Training")
        #self.model.train()
        for key, model in self.model.items():
            model.train()
        epoch_loss = []
    
        total_batches = len(self.trainLoader)
        print("=====> the number of iterations per epoch: ", total_batches)
        st = time.time()
        for iteration, batch in enumerate(self.trainLoader, 0):
    
            self.opts.per_iter = total_batches
            self.opts.max_iter = self.opts.max_epochs * self.opts.per_iter
            self.opts.cur_iter = self.epoch * self.opts.per_iter + iteration
            # learming scheduling
            if self.opts.lr_schedule == 'poly':
                lambda1 = lambda epoch: math.pow((1 - (self.opts.cur_iter / self.opts.max_iter)), self.opts.poly_exp)
                scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
            elif self.opts.lr_schedule == 'warmpoly':
                scheduler = WarmupPolyLR(self.optimizer, T_max=self.opts.max_iter, cur_iter=self.opts.cur_iter, warmup_factor=1.0 / 3,
                                     warmup_iters=self.opts.warmup_iters, power=0.9)
    
            lr = self.optimizer.param_groups[0]['lr']
    
            start_time = time.time()
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()
    
            #output = self.model(images)
            features = self.model['encoder'](images)
            if self.opts.feature_fusing == True:
                x_o = self.model['mono_encoder'](images)
                output = self.model['decoder'](features,x_o)
            else:
                output = self.model['decoder'](features)
###############################################################
            loss = self.criteria(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step() # In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
            epoch_loss.append(loss.item())
            time_taken = time.time() - start_time
    
            print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (self.epoch + 1, self.opts.max_epochs,iteration + 1, total_batches,lr, loss.item(), time_taken))
    
        time_taken_epoch = time.time() - st
        remain_time = time_taken_epoch * (self.opts.max_epochs - 1 - self.epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))
    
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    
        return average_epoch_loss_train, lr
  
    def val(self):
        """
        self.opts:
          val_loader: loaded for validation dataset
          model: model
        return: mean IoU and IoU class
        """
        # evaluation mode
        #self.model.eval()
        for key, model in self.model.items():
            model.eval()
        total_batches = len(self.valLoader)
    
        data_list = []
        for i, (input, label, size, name) in enumerate(self.valLoader):
            start_time = time.time()
            with torch.no_grad():
                # input_var = Variable(input).cuda()
                input_var = input.cuda()
                #output = self.model(input_var)
                feature = self.model['encoder'](input_var)
                output = self.model['decoder'](feature)
            time_taken = time.time() - start_time
            print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
            output = output.cpu().data[0].numpy()
            gt = np.asarray(label[0].numpy(), dtype=np.uint8)
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            data_list.append([gt.flatten(), output.flatten()])
    
        meanIoU, per_class_iu = get_iou(data_list, self.opts.classes)
        return meanIoU, per_class_iu

    def save_model(self):
            
        #model_file_name = self.opts.savedir + '/model_' + str(self.epoch + 1) + '.pth'
        #state = {"epoch": self.epoch + 1, "model": self.model.state_dict()}
        
        # Individual Setting for save model !!!
        if self.opts.dataset == 'camvid':
            torch.save(state, model_file_name)
        elif self.opts.dataset == 'cityscapes':
            if self.epoch >= self.opts.max_epochs - 10:
                for model_name, model in self.model.items():
                    save_path = os.path.join(self.opts.savedir + '{}_{}'.format(model_name,self.epoch) + '.pth')
                    to_save = model.state_dict()
                    torch.save(to_save, save_path)
            elif not self.epoch % 50:
                for model_name, model in self.model.items():
                    save_path = os.path.join(self.opts.savedir + '{}_{}'.format(model_name,self.epoch) + '.pth')
                    to_save = model.state_dict()
                    torch.save(to_save, save_path)
                #torch.save(state, model_file_name)

        # draw plots for visualization
        if self.epoch % 50 == 0 or self.epoch == (self.opts.max_epochs - 1):
            # Plot the figures per 50 epochs
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(self.start_epoch, self.epoch + 1), self.lossTr_list)
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")

            plt.savefig(self.opts.savedir + "loss_vs_epochs.png")
            plt.clf()
            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(self.epochs_list, self.mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            plt.legend(loc='lower right')

            plt.savefig(self.opts.savedir + "iou_vs_epochs.png")
            plt.close('all')
    
    if __name__ == '__main__':
        start = timeit.default_timer()
        if self.opts.dataset == 'cityscapes':
            self.opts.classes = 19
            self.opts.input_size = '512,1024'
            ignore_label = 255
        elif self.opts.dataset == 'camvid':
            self.opts.classes = 11
            self.opts.input_size = '360,480'
            ignore_label = 11
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % self.opts.dataset)
         
        train_model(self.opts)
        end = timeit.default_timer()
        hour = 1.0 * (end - start) / 3600
        minute = (hour - int(hour)) * 60
        print("training time: %d hour %d minutes" % (int(hour), int(minute)))
