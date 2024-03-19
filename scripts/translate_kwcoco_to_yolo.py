import json
import kwcoco
import os
import argparse
from glob import glob
import ubelt as ub
import shutil
from pathlib import Path
from PIL import Image


"""
python scripts/translate_kwcoco_to_yolo.py --dset /data/PTG/medical/object_anns/m5/v0.56/M3_Pressure_Dressing_YoloModel_LO_train.mscoco.json --output_dir /data/PTG/medical/object_anns/m5/v0.56/ --split train --ignore_classes hand
python scripts/translate_kwcoco_to_yolo.py --output_dir /data/PTG/medical/object_anns/m5/v0.56/ --ignore_classes hand


"""

SPLITS = ["train", "test"]

def dictionary_contents(path: str, types: list, recursive: bool = False) -> list:
    """
    Extract files of specified types from directories, optionally recursively.

    Parameters:
        path (str): Root directory path.
        types (list): List of file types (extensions) to be extracted.
        recursive (bool, optional): Search for files in subsequent directories if True. Default is False.

    Returns:
        list: List of file paths with full paths.
    """
    files = []
    if recursive:
        path = path + "/**/*"
    for type in types:
        if recursive:
            for x in glob(path + type, recursive=True):
                files.append(os.path.join(path, x))
        else:
            for x in glob(path + type):
                files.append(os.path.join(path, x))
    return files


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
    dataset_paths = dictionary_contents(output_dir, types=["*.mscoco.json"])
    splits_paths = dictionary_contents(output_dir, types=["*.text"])
    
    labels_dir = f"{output_dir}/labels"
    if len(splits_paths)>0:
        if os.path.exists(labels_dir):
            shutil.rmtree(labels_dir)
        for split_file in splits_paths:
            if os.path.exists(split_file):
                os.remove(split_file)
    
    for split in SPLITS:
        split_labels_path = f"{labels_dir}/{split}/"
        if not os.path.exists(split_labels_path):
            os.mkdir(split_labels_path)
            
        ignored_objs = 0
        ignored_images = 0
        
        for dataset_path in dataset_paths:
            if split in dataset_path:
                dset = dataset_path
                # print(f"dataset paths: {dataset_path}")
        # exit()
        dset = load_kwcoco(dset)
        old_to_new_cid_mapping = {}
        rm_cat_ids = []
        affected_ids = []
        max_id = 0
        for index, item in enumerate(dset.dataset['categories']):
            id, cat = item['id'], item['name']
            if int(id)> max_id:
                max_id = int(id)
            
            if cat in ignore_classes:
                rm_cat_ids.append(id)
                affected_ids.append((id, dset.dataset['categories'][index+1:]))
                
        
        # exit()
        # print(f"dset cats: {dset.dataset['categories']}")
        # print(f"rm_cat_ids: {rm_cat_ids}")
        print(f"affected_ids: {affected_ids}")
        
        for affected_id in affected_ids:
            rm = affected_id[0]
            affected = affected_id[1]
            for aff in affected:
                id = int(aff['id'])
                sub = id - rm
                new_id = id - sub
                # print(f"new id: {new_id}")
                old_to_new_cid_mapping[id] = new_id
                # old_to_new_cid_mapping[id] = id
        
        remove_info = dset.remove_categories(cat_identifiers=ignore_classes)
        # print(f"dset cats: {dset.dataset['categories']}")
        print(f"remove_info: {remove_info}")
        print(f"old_to_new_cid_mapping: {old_to_new_cid_mapping}")
        # exit()
        
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

                        cat_id = obj['category_id']
                        if cat_id in old_to_new_cid_mapping.keys():
                            new_cat_id = old_to_new_cid_mapping[cat_id]
                        else:
                            new_cat_id = cat_id
                            
                        # cls_name = dset.cats[new_cat_id]["name"]

                        # if cls_name in ignore_classes:
                        #     ignored_objs += 1
                        #     continue
                        
                        print(f"cat id saved: {new_cat_id}")
                        # text_file.write(f"{obj['category_id']} ")
                        text_file.write(f"{new_cat_id} ")
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
        "--dir",
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
