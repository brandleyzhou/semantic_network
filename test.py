import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict

### network selections####

#from model.ENet_network import encoder
#from model.ENet_network import decoder

from model import hr_networks
def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', type=str, default="ENet", choices = ['MONO','FPENet', 'SQNet', 'ENet'], help="model name: (default ENet)")
    parser.add_argument('--feature_fusing', type=bool, default = False, help="decide whether to fuse features")
    parser.add_argument('--dataset', default="cityscapes", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--best', action='store_true', help="Get the best result among last few checkpoints")
    parser.add_argument("--epoch", type=int, help="number of resnet layers", default=0)
    parser.add_argument("--num_layers", type=int, help="number of resnet layers", default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--save', action='store_true', help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args


def test(args, test_loader):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    device = 'cuda' 
    # evaluation or test mode
    print(device)
    model_path = args.checkpoint
    print("-> loading model from", model_path)
    encoder_path = os.path.join(model_path, "encoder_{}.pth".format(args.epoch))
    decoder_path = os.path.join(model_path, "decoder_{}.pth".format(args.epoch))

######depth_encoder
    if args.feature_fusing == True:
        depth_encoder = args.depth_encoder
        depth_encoder_path = 'model/mono_encoder.pth'
        loaded_dict_enc = torch.load(depth_encoder_path, map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
        depth_encoder.load_state_dict(filtered_dict_enc)
        #depth_encoder.load_state_dict(loaded_dict_enc)
        depth_encoder.to(device)
        depth_encoder.eval()

######encoder
    encoder = args.encoder
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    #encoder.load_state_dict(loaded_dict_enc)
    encoder.cuda()
    encoder.eval()

######decoder
    decoder = args.decoder
    loaded_dict = torch.load(decoder_path, map_location=device)
    decoder.load_state_dict(loaded_dict)
    decoder.cuda()
    decoder.eval()
    
    total_batches = len(test_loader)
    data_list = []
    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
        start_time = time.time()
        if args.feature_fusing == True:
            print(args.feature_fusing)
            features = encoder(input_var)
            x_o = depth_encoder(input_var)
            print(features[0].get_device())
            print('s',x_o.get_device())
            output = decoder(features, x_o)
        else:
            output = decoder(encoder(input_var))
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

        # save the predicted image
        if args.save:
            save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=True)

    meanIoU, per_class_iu = get_iou(data_list, args.classes)
    return meanIoU, per_class_iu


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    #model = build_model(args.model, num_classes=args.classes)

    #if args.cuda:
    #    model = model.cuda()  # using GPU for inference
    #    cudnn.benchmark = True

    #if args.save:
    #    if not os.path.exists(args.save_seg_dir):
    #        os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    if not args.best:
        print("=====> beginning validation")
        print("validation set length: ", len(testLoader))
        mIOU_val, per_class_iu = test(args, testLoader)
        print(mIOU_val)
        print(per_class_iu)

    # Get the best test result among the last 10 model records.

    # Save the result
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        args.logFile = 'test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Mean IoU: %.4f" % mIOU_val)
        logger.write("\nPer class IoU: ")
        for i in range(len(per_class_iu)):
            logger.write("%.4f\t" % per_class_iu[i])
    logger.flush()
    logger.close()


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)
    args.classes = 19
    if args.model == 'ENet':
        from model.ENet_network import encoder, decoder
        args.encoder = encoder.ENet_Encoder()
        args.decoder = decoder.ENet_Decoder()
    elif args.model == 'FPENet':
        from model.FPENet_network import encoder,decoder
        self.model['encoder'] = encoder.FPENet_Encoder()
        self.model['decoder'] = decoder.FPENet_Decoder()
    else:
        print('No modelk')
    if args.feature_fusing == True:
        args.depth_encoder = hr_networks.ResnetEncoder(args.num_layers, True)
        #args.depth_decoder = hr_networks.DepthDecoder(num_ch_enc=args.encoder.num_ch_enc)
    
    test_model(args)
