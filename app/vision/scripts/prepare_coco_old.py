from pycocotools.coco import COCO
# from datasets import build_dataset
from PIL import Image
import argparse
import os
import torch

def create_training_list(img_folder, ann_file, output_file):
    coco = COCO(ann_file)
    ids = list(sorted(coco.imgs.keys()))

    i = 0;
    with open(output_file, 'w') as out:
        for idx in ids:
            filepath = coco.loadImgs(idx)[0]['file_name']
            img = Image.open(os.path.join(img_folder, filepath))
            w, h = img.size
            ann_ids = coco.getAnnIds(idx)
            anns = coco.loadAnns(ann_ids)
            bboxes = [ ann['bbox'] for ann in anns]
            labels = [ ann['category_id'] for ann in anns]
            # boxes = torch.as_tensor(bboxes).reshape(-1, 4);
            boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            if len(boxes) == 0:
                print('no boxes!')
                continue
            # import pdb; pdb.set_trace()
            labels = labels[keep]
            labels = labels.tolist()
            out.write(f'{os.path.join(img_folder, filepath)}\t')
            # print(boxes)
            # print(labels)
            strings = []
            for (box, label) in zip(boxes, labels):
                box_with_label = box.tolist() + [ label ]
                # print(box)
                box_string = " ".join(map(str, box_with_label));
                strings.append(box_string)
                # print(box_string)
            out.write(" ".join(strings))
            out.write('\n')
            i += 1
            if i == 128:
                break;

# def dump_dataset(image_set, image_folder, args):
    # dataset = build_dataset(image_set, args);
    # output_file = os.path.join(args.output_dir, f'{image_set}.lst')
    # i = 0
    # with open(output_file, 'w') as out:
        # for sample in dataset:
            # (_, targets) = sample
            # bboxes = targets["boxes"]
            # labels = targets["labels"]
            # # print('bboxes', bboxes)
            # # print('labels', labels)
            # image_id = targets["image_id"].item();
            # filepath = f'{image_id:012}.jpg'
            # out.write(f'{os.path.join(img_folder, filepath)}\t')
            # # Tabs seperate boxes
            # strings = []
            # for (bbox, label) in zip(bboxes, labels):
                # box_with_label = bbox.tolist() + [ label.item() ];
                # box_string = " ".join(map(str, box_with_label));
                # strings.append(box_string);
            # # print(strings)
            # out.write(" ".join(strings))
            # out.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--crowdfree', action='store_true',
                        help='Remove crowd images from training on COCO')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('-r', '--root', help='Root of COCO data', default='/datasets01/COCO/022719')
    parser.add_argument('-o', '--output_dir', help='Output dir .lst file', default='/private/home/padentomasello/data/coco-mini2/')
    args = parser.parse_args()

    # dataset_train = dump_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='test', args=args)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--root', help='Root of COCO data', default='/datasets01/COCO/022719')
    # parser.add_argument('-o', '--output_dir', help='Output dir .lst file', default='/private/home/padentomasello/data/coco/')
    # args = parser.parse_args()
    root = args.root

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

