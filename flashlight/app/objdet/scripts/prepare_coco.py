"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from pycocotools.coco import COCO
from PIL import Image
import argparse
import os
import torch

def create_training_list(img_folder, ann_file, output_file):
    coco = COCO(ann_file)
    ids = sorted(coco.imgs.keys())

    with open(output_file, 'w') as out:
        for idx in ids:
            filepath = coco.loadImgs(idx)[0]['file_name']
            img = Image.open(os.path.join(img_folder, filepath))
            w, h = img.size
            ann_ids = coco.getAnnIds(idx)
            anns = coco.loadAnns(ann_ids)
            bboxes = [ ann['bbox'] for ann in anns]
            labels = [ ann['category_id'] for ann in anns]
            boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = labels[keep]
            labels = labels.tolist()
            filepath = os.path.join(img_folder, filepath);
            strings = [filepath]
            for (box, label) in zip(boxes, labels):
                box_with_label = box.tolist() + [ label ]
                box_string = " ".join(map(str, box_with_label));
                strings.append(box_string)
            out.write("\t".join(strings))
            out.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--crowdfree', action='store_true',
                        help='Remove crowd images from training on COCO')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('-r', '--coco_path', help='Root of COCO data', default='/datasets01/COCO/022719')
    parser.add_argument('-o', '--output_dir', help='Output dir .lst file', default='/private/home/padentomasello/data/coco-mini/')
    args = parser.parse_args()

    root = args.coco_path

    anno_file_template = "instances_{}2017.json"
    # # Directory of Split -> (img_folder, annotation file)
    paths = {
        "train": ("train2017", anno_file_template.format("train")),
        "val": ("val2017", anno_file_template.format("val")),
    }
    for (split, (img_folder, ann_file)) in paths.items():
        img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, 'annotations', ann_file)
        output_file = os.path.join(args.output_dir, split + '.lst')
        create_training_list(img_folder, ann_file, output_file)


