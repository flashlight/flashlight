import argparse
import arrayfire as af
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import build as build_coco
from datasets.coco_eval import CocoEvaluator
import util.box_ops
import glob
import os

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


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
        boxes = util.box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def prepare_for_coco_detection(predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

class Args(object):

    coco_path = '/datasets01/COCO/022719'
    masks = False

def main(directory):

    args = Args()

    # dataset_val = build_coco(image_set='val', args=args)
    # dataset_val = build_coco(image_set='val', args=args)
    # base_ds = get_coco_api_from_dataset(dataset_val)
    # coco_evaluator = CocoEvaluator(base_ds, ('bbox',))

    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    dataDir='/datasets01/COCO/022719/'
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)


    all_results = []
    all_image_ids = []
    # imageIds = [f'/datasets01/COCO/022719/train2017/{id:012d}.jpg' for id in imageIds]

    postprocess = PostProcess();
    for f in glob.glob(os.path.join(directory, 'detection*.array')):

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

        res = { id : output for id, output in zip(imageIds, results) };
        results = prepare_for_coco_detection(res)

        imageIds = [ id for id in imageIds ];

        all_results.extend(results)
        all_image_ids.extend(imageIds)
        # print(imageIds)

    cocoDt = coco.loadRes(all_results)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
#cocoEval.params.imgIds  = [imageId]
    cocoEval.params.imgIds  = all_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
        # res = { id : output for id, output in zip(imageIds, results) };
        # results = prepare_for_coco_detection(res)

    # print(len(coco_evaluator.img_ids))
    # print(len(coco_evaluator.eval_imgs))
    # coco_evaluator.synchronize_between_processes()
    # print(len(coco_evaluator.img_ids))
    # print(len(coco_evaluator.eval_imgs))
    # # coco_evaluator.coco_eval['bbox'].eval_imgs = imageIds
    # coco_evaluator.coco_eval['bbox'].params.imgIds = all_ids
    # # coco_evaluator.eval_imgs['bbox'] = imageIds;
    # print(len(coco_evaluator.img_ids))
    # print(len(coco_evaluator.eval_imgs))
    # # coco_evaluator.evalImgs = all_ids
    # # coco_evaluator.evalImgs = all_ids
    # coco_evaluator.accumulate()
    # coco_evaluator.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='/private/home/padentomasello/data/coco/output/')
    args = parser.parse_args()
    main(args.dir)



