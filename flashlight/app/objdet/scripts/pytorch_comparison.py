import torch
import torchvision
import argparse
import arrayfire as af
import array


def toArrayFire(x):
    x_np = x.detach().contiguous().numpy()
    shape = 1
    if len(x_np.shape) == 0:
        shape = (1,)
    else:
        shape = x_np.shape[::-1]
    afArray = af.Array(x_np.ctypes.data, shape, x_np.dtype.char)
    return afArray

def saveStateDict(model, filepath):
    params = {}
    i = 0
    for (name, param) in model.state_dict().items():
        if 'running' in name:
            continue
        if 'in_proj' in name:
            q, k, v = param.chunk(3, dim=0)
            hack = '0'
            if 'in_proj_bias' in name: hack = '1'
            params['0q_' + hack + name] = q
            params['1k_' + hack + name] = k
            params['2v_' + hack + name] = v
            if 'in_proj_bias' in name:
                for key in sorted(params.keys()):
                    af_array = toArrayFire(params[key])
                    if 'weight' in key:
                        af_array = af.array.transpose(af_array)
                    #print(key, i, params[key].shape)
                    #print(af.array.save_array(key, af_array, filepath, True))
                    i = i + 1
                params = {}
            continue
        elif len(param.size()) > 0:
            if 'input_proj.bias' in name:
                param = param.reshape((1, 1, 256))
            af_array = toArrayFire(param)
            if 'fc' in name and 'weight' in name:
                af_array = af.array.transpose(af_array)
            elif 'weight' in name and 'proj' in name:
                af_array = af.array.transpose(af_array)
            elif 'weight' in name and 'linear' in name:
                af_array = af.array.transpose(af_array)
            elif 'query_embed' in name:
                af_array = af_array
            elif 'weight' in name and 'embed' in name:
                af_array = af.array.transpose(af_array)

            #print(name, i, param.shape)
            #print(af.array.save_array(name, af_array, filepath, True))
            i = i + 1
    for name in model.state_dict():
        if 'running' in name:
            print(name)
            af_array = toArrayFire(model.state_dict()[name])
            print(name, model.state_dict()[name].shape, af.array.save_array(name, af_array, filepath + 'running', True))
    
def create_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--optimizer', default="adam", type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval_skip', default=1, type=int,
                        help='do evaluation every "eval_skip" frames')
    parser.add_argument('--schedule', default='step', type=str,
                        choices=('step', 'multistep'))

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--no_pass_pos_and_query', dest='pass_pos_and_query', action='store_false',
                        help="Disables passing the positional encodings to each attention layers")

    # * Segmentation
    parser.add_argument('--mask_model', default='none', type=str, choices=("none", "smallconv", "v2"),
                        help="Segmentation head to be used (if None, segmentation will not be trained)")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_loss', default='hungarian', type=str,
                        choices=('sequential', 'hungarian', 'lexicographical'),
                        help="Type of matching to perform in the loss")
    parser.add_argument('--bcl', dest='use_bcl', action='store_true',
                        help="Use balanced classification loss")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/datasets01/COCO/022719')
    parser.add_argument('--coco_panoptic_path', type=str, default='/datasets01/COCO/060419')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--masks', action='store_true')

    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return parser

import models.detr
import datasets.coco
parser = create_parser()
pretrained_path = '/private/home/padentomasello/scratch/pytorch_testing/detr-r50-e632da11.pth'
args = parser.parse_args(["--resume=/private/home/padentomasello/scratch/pytorch_testing/detr-r50-e632da11.pth"])
args = parser.parse_args([])
model, criterion, post = models.detr.build(args)
dataset = datasets.coco.build('train', args)
if args.resume:
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])


from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
dataset_train = dataset
sampler_train = torch.utils.data.SequentialSampler(dataset)
#batch_sampler_train = torch.utils.data.BatchSampler(
    #    sampler_train, args.batch_size, drop_last=True)
batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 1, drop_last=True)
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                            collate_fn=utils.collate_fn, num_workers=1)


for tmp in data_loader_train:
    sample = tmp
    break

from models.transformer import *


from models.backbone import *
filepath = '/private/home/padentomasello/scratch/pytorch_testing/detr.array'

N = 2
C = 3
H = 224
W = 224

embedding_size = 8
tgt_len = 10

queries = torch.rand(tgt_len, embedding_size)
image = sample[0].tensors
mask = sample[0].mask

af.array.save_array('image', toArrayFire(image), filepath, False)
#af.array.save_array('queries', toArrayFire(queries), filepath, True)
af.array.save_array('mask', toArrayFire(mask.float()), filepath, True)
#af.array.save_array('pos', toArrayFire(pos), filepath, True)
       


model.eval()
output = model(sample[0])
saveStateDict(model, filepath)
af.array.save_array('pred_logits', toArrayFire(output['pred_logits']), filepath, True)
af.array.save_array('pred_boxes', toArrayFire(output['pred_boxes']), filepath, True)

criterion(output, sample[1])
