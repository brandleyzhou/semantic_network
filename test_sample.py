import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms
from argparse import ArgumentParser
# user
import cv2
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.convert_state import convert_state_dict
#from model import hr_networks
from model.ENet_network import encoder
from model.ENet_network import decoder


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', default="ENet", help="model name: (default ENet)")
    parser.add_argument("--num_layers", type=int, help="number of resnet layers", default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument("--epoch", type=int, help="number of resnet layers", default=200)
    parser.add_argument('--image_path', help="image_path")
    parser.add_argument('--num_workers', type=int, default=2, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str, help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./results/",
                        help="saving path of prediction result")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args

def predict(args):
    """
    args:
      test_image: loaded for test, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
###################Load pretrained models#########################################
    if args.cuda == True:
        device = 'cuda'
    else:
        device = 'cpu'

    #model_path = os.path.join(args.model_folder, args.model_name)
    model_path = args.checkpoint
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder_{}.pth".format(args.epoch))
    decoder_path = os.path.join(model_path, "decoder_{}.pth".format(args.epoch))
    
    encoder = args.encoder.ENet_Encoder()
    #encoder = hr_networks.ResnetEncoder(args.num_layers, True)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    #decoder = hr_networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc)
    decoder = args.decoder.ENet_Decoder()
    loaded_dict = torch.load(decoder_path, map_location=device)
    decoder.load_state_dict(loaded_dict)
    decoder.to(device)
    decoder.eval()

####################################################################################
    with torch.no_grad():
        input_image = cv2.imread(args.image_path,cv2.IMREAD_COLOR)
        #input_image = cv2.resize(input_image, (256,128))
        print(input_image.shape)
        input_image = np.asarray(input_image, np.float32)
        #input_image = np.asarray(input_image)
        
        #without this getting better visualization
        mean = (128, 128, 128) 
        input_image -= mean
                
        input_image = input_image[:, :, ::-1]  # change to RGB
        input_image = input_image.copy()
        original_width,original_height,_ = input_image.shape
        print(original_width,original_height)
        input_image = transforms.ToTensor()(input_image).to(device) 
        input_image = input_image.unsqueeze(0)
        output = decoder(encoder(input_image))
        #torch.cuda.synchronize()
        #output = output.squeeze(0).cpu().data[0].numpy()
        output = output.squeeze(0).cpu().numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis = 2), dtype=np.uint8)
        #Save the predict greyscale output for Cityscapes official evaluation
        #Modify image name to meet official requirement

        #save_predict(output, None,'seg', args.image_path, args.save_seg_dir,
        #         output_grey=True, output_color=False, gt_color=False)
        save_predict(output, None, 'seg', 'cityscapes', args.save_seg_dir,
                 output_grey=False, output_color= True, gt_color=False)

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

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    #datas, testLoader = build_dataset_test(args.dataset, args.num_workers, none_gt=True)

    #if args.checkpoint:
    #    if os.path.isfile(args.checkpoint):
    #        print("=====> loading checkpoint '{}'".format(args.checkpoint))
    #        checkpoint = torch.load(args.checkpoint)
    #        model.load_state_dict(checkpoint['model'])
    #        # model.load_state_dict(convert_state_dict(checkpoint['model']))
    #    else:
    #        print("=====> no checkpoint found at '{}'".format(args.checkpoint))
    #        raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning testing")
    #print("test set length: ", len(testLoader))
    #predict(args, model)
    predict(args)

if __name__ == '__main__':

    args = parse_args()
    args.save_seg_dir = os.path.join(args.save_seg_dir, args.image_path, 'predict', args.model)
    args.classes = 19
    args.encoder = encoder
    args.decoder = decoder
    test_model(args)
