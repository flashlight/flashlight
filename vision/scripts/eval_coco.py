import arrayfire as af
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import build as build_coco
from datasets.coco_eval import CocoEvaluator
import box_ops
import glob


# from datasets.coco_eval import CocoEvaluator

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, out_logits, out_bbox, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class Args(object):

    coco_path = '/datasets01/COCO/022719'
    masks = False

args = Args()

# dataset_val = build_coco(image_set='val', args=args)
dataset_val = build_coco(image_set='val', args=args)
base_ds = get_coco_api_from_dataset(dataset_val)
coco_evaluator = CocoEvaluator(base_ds, ('bbox',))

# imageIds = [f'/datasets01/COCO/022719/train2017/{id:012d}.jpg' for id in imageIds]

postprocess = PostProcess();
for f in glob.glob('/private/home/padentomasello/data/coco/output/detection*.array'):

    imageSizes = af.read_array(f, key='imageSizes').to_ndarray()
    imageSizes = np.transpose(imageSizes, (1, 0))
    imageSizes = torch.from_numpy(imageSizes)
    imageIds = af.read_array(f, key='imageIds').to_ndarray()
    # imageIds = np.transpose(imageIds, (1, 0))
    scores = af.read_array(f, key='scores').to_ndarray()
    scores = np.transpose(scores, (2, 1, 0))
    scores = torch.from_numpy(scores)
    bboxes = af.read_array(f, key='bboxes').to_ndarray()
    bboxes = np.transpose(bboxes, (2, 1, 0))
    bboxes = torch.from_numpy(bboxes)
    results = postprocess.forward(scores, bboxes, imageSizes)
    imageIds = [ id for id in imageIds ];
    # print(imageIds)

    res = { id : output for id, output in zip(imageIds, results) };
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()



