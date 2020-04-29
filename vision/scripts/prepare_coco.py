from pycocotools.coco import COCO
import argparse
import os

def create_training_list(img_folder, ann_file, output_file):
    coco = COCO(ann_file)
    ids = list(sorted(coco.imgs.keys()))

    with open(output_file, 'w') as out:
        for idx in ids:
            filepath = coco.loadImgs(idx)[0]['file_name']
            ann_ids = coco.getAnnIds(idx)
            anns = coco.loadAnns(ann_ids)
            bboxes = [ ann['bbox'] + [ float(ann['category_id']) ] for ann in anns]
            out.write(f'{os.path.join(img_folder, filepath)}\t')
            for box in bboxes:
                print(box)
                box_string = " ".join(map(str, box));
                out.write(f'{box_string}')
                break;
            out.write('\n')
            break;

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', help='Root of COCO data', default='/datasets01/COCO/022719')
    parser.add_argument('-o', '--output_dir', help='Output dir .lst file', default='/private/home/padentomasello/data/coco/')
    args = parser.parse_args()
    root = args.root

    anno_file_template = "instances_{}2017.json"
    # Directory of Split -> (img_folder, annotation file)
    paths = {
        "train": ("train2017", anno_file_template.format("train")),
        "val": ("val2017", anno_file_template.format("val")),
    }
    for (split, (img_folder, ann_file)) in paths.items():
        img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, "annotations", ann_file)
        output_file = os.path.join(args.output_dir, f'{split}.lst')
        create_training_list(img_folder, ann_file, output_file)

