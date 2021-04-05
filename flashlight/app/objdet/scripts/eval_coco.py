"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import arrayfire as af
import glob
import numpy as np
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = np.split(boxes, 4, axis=1)
    result = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=1);
    return result

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    max_value = np.max(x, axis=axis, keepdims=True);
    e_x = np.exp(x - max_value)
    s = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / s

def cxcywh_to_xyxy(x):
    x_c, y_c, w, h = np.split(x, 4, 2)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.concatenate(b, axis=2)

def postprocess_fn(out_logits, out_bbox, target_sizes):
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = softmax(out_logits, 2)
    labels = np.argmax(prob[..., :-1], axis=2);
    scores = np.amax(prob[..., :-1], axis=2)
    boxes = cxcywh_to_xyxy(out_bbox)

    img_h, img_w = np.split(target_sizes, 2, axis=1)

    scale_fct = np.concatenate([img_w, img_h, img_w, img_h], axis=1);
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results;



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


def main(directory, coco_data):
    data_type='val2017'
    ann_file='{}/annotations/instances_{}.json'.format(coco_data, data_type)
    coco=COCO(ann_file)


    all_results = []
    all_image_ids = []

    glob_path = os.path.join(directory, '**', 'detection*.array')
    files = glob.glob(glob_path)
    assert(len(files) > 0)
    for f in files:

        image_sizes = af.read_array(f, key='imageSizes').to_ndarray()
        image_sizes = np.transpose(image_sizes, (1, 0))
        image_ids = af.read_array(f, key='imageIds').to_ndarray()
        scores = af.read_array(f, key='scores').to_ndarray()
        scores = np.transpose(scores, (2, 1, 0))
        bboxes = af.read_array(f, key='bboxes').to_ndarray()
        bboxes = np.transpose(bboxes, (2, 1, 0))

        results = postprocess_fn(scores, bboxes, image_sizes)

        res = { id : output for id, output in zip(image_ids, results) };
        results = prepare_for_coco_detection(res)


        image_ids = [ id for id in image_ids ];

        all_results.extend(results)
        all_image_ids.extend(image_ids)

    coco_dt = coco.loadRes(all_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.imgIds  = all_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='/private/home/padentomasello/data/coco/output/')
    parser.add_argument('--coco', default='/datasets01/COCO/022719/')
    args = parser.parse_args()
    main(args.dir, args.coco)



