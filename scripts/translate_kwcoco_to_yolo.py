import json
import kwcoco
import os
import argparse

import ubelt as ub

from pathlib import Path
from PIL import Image


def load_kwcoco(dset):
    """Load a kwcoco dataset from file

    :param dset: kwcoco object or a string pointing to a kwcoco file

    :return: The loaded kwcoco object
    :rtype: kwcoco.CocoDataset
    """
    # Load kwcoco file
    if type(dset) == str:
        dset_fn = dset
        dset = kwcoco.CocoDataset(dset_fn)
        dset.fpath = dset_fn
        print(f"Loaded dset from file: {dset_fn}")
    return dset

def kwcoco_to_yolo(dset, output_dir, split, ignore_classes=[]):
    if ignore_classes is None:
        ignore_classes = []
    ignored_objs = 0
    ignored_images = 0
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))
    with open(f"{output_dir}/{split}.txt", "w") as split_file:
        for gid in gids:
            im = dset.imgs[gid]

            im_id = im['id']
            im_fname = im['file_name']

            # Check image size 
            #image = Image.open(im_fname)
            #w, h = image.size
            #image_size = f"{w}x{h}"

            #if image_size != "1280x720":
            #    ignored_images += 1
            #    continue
            
            split_file.write(f"{im_fname}\n")

            # Get all objects for this image
            aids = gid_to_aids[gid]
            anns = ub.dict_subset(dset.anns, aids)
            im_objects = list(anns.values()) #[ann for ann in dset['annotations'] if ann['image_id'] == im_id]

            im_fpath = f"{output_dir}/labels/{split}/" + os.path.splitext(os.path.basename(im_fname))[0] + ".txt"

            with open(im_fpath, "w") as text_file:
                for obj in im_objects:
                    x,y,w,h = obj['bbox']
                    x /= im["width"]
                    w /= im["width"]
                    y /= im["height"]
                    h /= im["height"]

                    cls_name = dset.cats[obj['category_id']]["name"]

                    if cls_name in ignore_classes:
                        ignored_objs += 1
                        continue

                    text_file.write(f"{obj['category_id']} ")
                    text_file.write(f"{x} {y} {x+w} {y} {x+w} {y+h} {x} {y+h}\n")

    print(f"Ignored {ignored_objs} objects with labels {ignore_classes}")
    print(f"Ignored {ignored_images} images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset",
        type=str,
        default="train.mscoco.json",
        help="Kwcoco dataset to convert",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Output folder",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--ignore_classes",
        type=str,
        nargs='+',
        help="Classes to ignore in the resulting dataset",
    )

    args = parser.parse_args()

    Path(f"{args.output_dir}/labels/{args.split}").mkdir(parents=True, exist_ok=True)

    kwcoco_to_yolo(args.dset, args.output_dir, args.split, args.ignore_classes)


if __name__ == "__main__":
    main()
