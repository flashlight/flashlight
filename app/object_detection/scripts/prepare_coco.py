from pycocotools.coco import COCO
from datasets import build_dataset
import argparse
import os

# def create_training_list(img_folder, ann_file, output_file):
    # coco = COCO(ann_file)
    # ids = list(sorted(coco.imgs.keys()))

    # with open(output_file, 'w') as out:
        # for idx in ids:
            # filepath = coco.loadImgs(idx)[0]['file_name']
            # ann_ids = coco.getAnnIds(idx)
            # anns = coco.loadAnns(ann_ids)
            # bboxes = [ ann['bbox'] + [ float(ann['category_id']) ] for ann in anns]
            # out.write(f'{os.path.join(img_folder, filepath)}\t')
            # for box in bboxes:
                # print(box)
                # box_string = " ".join(map(str, box));
                # out.write(f'{box_string}')
                # break;
            # out.write('\n')
            # break;

def dump_dataset(image_set, image_folder, args):
    dataset = build_dataset(image_set, args);
    output_file = os.path.join(args.output_dir, f'{image_set}.lst')
    i = 0
    with open(output_file, 'w') as out:
        for sample in dataset:
            (_, targets) = sample
            bboxes = targets["boxes"]
            print(bboxes)
            break;
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
    parser.add_argument('-o', '--output_dir', help='Output dir .lst file', default='/private/home/padentomasello/data/coco-mini/')
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
        dump_dataset(split, img_folder, args)

