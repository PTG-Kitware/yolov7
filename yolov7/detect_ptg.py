#!/usr/bin/env python3

from pathlib import Path
import random
from typing import List, Optional, Tuple, Union

import click
import cv2
import kwcoco
import moviepy.video.io.ImageSequenceClip
import numpy as np
import numpy.typing as npt
import torch
import ubelt as ub
from ultralytics import YOLO
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from yolov7.utils.torch_utils import select_device, TracedModel
from yolov7.utils.plots import plot_one_box
from yolov7.utils.datasets import letterbox


"""

python yolov7/detect_ptg.py \
--tasks m2 \
--split val \
--weights /data/PTG/medical/training/yolo_object_detector/train/m2_good_objects_only_no_hand_no_amputation_stump_hue/weights/best.pt \
--project /data/PTG/medical/training/yolo_object_detector/detect/ \
--name m2_good_objects_only_no_hand_no_amputation_stump_hue_v2 \
--device 0 \
--save-img \
--img-size 768 \
--conf-thres 0.1

python yolov7/detect_ptg.py \
--tasks m2 \
--split val \
--weights /data/PTG/medical/training/yolo_object_detector/train/m2_good_objects_only_no_hand_no_amputation_stump_hue/weights/best.pt \
--project /data/PTG/medical/training/yolo_object_detector/detect/ \
--name m2_good_objects_only_no_hand_no_amputation_stump_hue_v2 --device 0 --save-img \
--img-size 768 \ 
--conf-thres 0.1

python yolov7/detect_ptg.py \
--tasks m3 \
--split train \
--weights /data/PTG/medical/training/yolo_object_detector/train/m3_all_v16/weights/best.pt \
--project /data/PTG/medical/training/yolo_object_detector/detect/ \
--name m3_all \
--device 0 \
--save-img \
--img-size 768 \
--conf-thres 0.25

"""


def preprocess_bgr_img(
    img_bgr: npt.ArrayLike,
    imgsz: Union[int, Tuple[int, int]],
    stride: int,
    device: torch.device,
    half: bool,
) -> torch.Tensor:
    """
    Prepare a BGR image matrix for inference processing by the model.

    :param img_bgr: Input BGR image matrix in [H, W, C] order.
    :param imgsz: Integer (square size) or [height, width] tuple specifying the
        target image size to transform to.
    :param stride: Model stride to take into account.
    :param device: Device the model is located on so we can move our tensor
        there as well.
    :param half: True if the model is using half width (FP16), false if not
        (FP32).

    :return: The processed image as a Tensor instance on the same device as
        input.
    """
    # Padded resize
    img = letterbox(img_bgr, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Add the batch dimension if it is not there.
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def load_model(device, weights_fp, img_size):
    """
    Load Yolo v7 model from provided weights.

    :param device: Device type to load the model and weights onto.
    :param weights_fp: Filesystem path to the weights to load.
    :param img_size: Desired network input size to be checked and adjusted.

    :return: This will return a tuple of:
        [0] Torch device instance the model is loaded onto.
        [1] The model instance.
        [2] The model's stride.
        [3] The **adjusted** image input size to be given to the loaded model.
    """
    device = select_device(device)
    model = attempt_load(weights_fp, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    return device, model, stride, imgsz


def predict_image(
    img0: npt.NDArray,
    device: torch.device,
    model: torch.nn.Module,
    # Preprocessing parameters
    stride: int,
    imgsz: Union[int, Tuple[int, int]],
    half: bool,
    # Prediction parameters
    augment: bool,
    # NMS Parameters
    det_conf_threshold: float,
    iou_threshold: float,
    filter_classes: Optional[List[int]],
    class_agnostic_nms: bool,
):
    """
    Perform a full prediction step given a raw BGR input image.

    This function results in a generator that yields individual detection
    attributes as tuples.
    Each yielded element will be of the format:
        * [0] xyxy bounding-box 4-tensor (upper-left, lower-right) in input
          image coordinate space.
        * [1] float detection confidence score
        * [2] integer predicted class index (indexing `model.names`)

    Detection confidence filtering is based on the product of the detection
    score times the class score, occurring within the non-max-suppression
    operation.

    :param img0: Original input image matrix in [H, W, C] format.
    :param img: Pre-processed image as a torch.Tensor. Assumed to be on the
        same device as the model.
    :param imgsz: Integer (square size) or [height, width] tuple specifying the
        target image size to transform to.
    :param model: The model to perform inference with. Expecting a module with
        a `forward` method with the same interface as that in
        `yolov7.models.yolo.Model`.
    :param stride: Model stride to take into account.
    :param device: Device the model is located on, so we can move our tensor
        there as well.
    :param half: True if the model is using half width (FP16), false if not
        (FP32).
    :param augment: If augmentation should be applied for this inference.
    :param det_conf_threshold: Confidence threshold in the [0, 1] range that
        output detections must meet or exceed.
    :param iou_threshold: IoU threshold for non-max-suppression operation.
    :param filter_classes: List of classes to filter output detections -- only
        detections whose class is in this list passes the filter. Pass `None`
        to **not** filter detections by class. An empty list means all
        detections will be filtered **out** (nothing to match against).
    :param class_agnostic_nms: IDK, usually seen this be False.

    :return: Generator over individual detections.
    """
    height, width = img0.shape[:2]

    # Preprocess image
    img = preprocess_bgr_img(img0, imgsz, stride, device, half)

    # Predict
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        # Shape: [batch_size, n_dets, bbox(4)+conf(1)+class_conf(n_classes)]
        #                             \-> 5 + n_classes
        # If augment is true, the `n_dets` dimension of `pred` will be larger.
        pred = model(img, augment=augment)[0]

    # Apply NMS
    pred_nms = non_max_suppression(pred, det_conf_threshold, iou_threshold,
                                   classes=filter_classes,
                                   agnostic=class_agnostic_nms)
    assert len(pred_nms) == 1, (
        f"We are only processing one image, so only a list of 1 element "
        f"should only every be returned from non_max_suppression. "
        f"Instead received: {len(pred_nms)}"
    )

    # Post-process detections
    dets = pred_nms[0].cpu()

    # Rescale boxes from img_size to img0 size
    dets[:, :4] = scale_coords(img.shape[2:], dets[:, :4], img0.shape).round()

    # Chesterton's Fence: Why reversed?
    if dets.shape[0] == 0:
        return [], [], []
    else:
        boxes, confs, classids = [], [], []
        for *xyxy, conf, cls_id in reversed(dets):  # ul-xy, lr-xy
            boxes.append(torch.tensor(xyxy))
            confs.append(conf)
            classids.append(cls_id.int())
        return boxes, confs, classids


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i", "--input-coco-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "MS-COCO file specifying image files to perform object detection over. "
        "Image and Video sections from this COCO file will be maintained in the "
        "output COCO file."
    ),
    required=True,
)
@click.option(
    "--img-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional override for the input COCO dataset bundle root. This is "
        "necessary when the input COCO file uses relative paths and the COCO "
        "file itself is not located in the bundle root directory."
    ),
)
@click.option(
    "-o", "--output-coco-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output COCO file to write object detection results.",
    required=True,
)
@click.option(
    "--model-hands", "hand_model_ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint for the Yolo v8 hand detector.",
    required=True,
)
@click.option(
    "--model-objects", "objs_model_ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint for the Yolo v7 object detector.",
    required=True,
)
@click.option(
    "--model-device",
    default="",
    help="The CUDA device to use, i.e. '0' or '0,1,2,3' or 'cpu'."
)
@click.option(
    "--img-size",
    type=int,
    default=1280,
    help=(
        "Data input size for object detection models. This should be a "
        "multiple of the model's stride parameter."
    )
)
@click.option(
    "--no-trace",
    is_flag=True,
    help="Don't trace the Yolo v7 model.",
)
@click.option(
    "--augment", "inference_augment",
    is_flag=True,
    help="Augment data during inferencing."
)
@click.option(
    "--conf-thres",
    type=float,
    default=0.25,
    help=(
        "Object confidence threshold. Predicted objects with confidence less "
        "than this will not be considered for output."
    ),
)
@click.option(
    "--iou-thres",
    type=float,
    default=0.45,
    help=(
        "IoU threshold used during NMS to filter out overlapping bounding "
        "boxes."
    ),
)
@click.option(
    "--agnostic-nms",
    is_flag=True,
    help="Perform non-maximum suppression across all classes.",
)
@click.option(
    "--save-img", "save_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Optionally enable the plotting of detections back to the image and "
        "saving them out to disk, rooted in this directory. Only detections "
        "with confidence above our configured threshold will be considered "
        "for plotting."
    )
)
@click.option(
    "--top-k", "save_top_k",
    type=int,
    default=None,
    help=(
        "Optionally specify that only the top N confidence detections should "
        "be saved to the output images. If this is not provided, all "
        "detections with confidence above the --conf-thres value will be "
        "plotted. This only applies to objects, not detected hands by that "
        "respective model."
    )
)
@click.option(
    "--save-vid",
    is_flag=True,
    help=(
        "Optionally enable the creation of an MP4 video from the images "
        "rendered due to --save-img. This option only has an effect if the "
        "--save-img option is provided. The video file will be save next to "
        "the directory into which component images are saved."
    )
)
def detect_v2(
    input_coco_file: Path,
    img_root: Optional[Path],
    output_coco_file: Path,
    hand_model_ckpt: Path,
    objs_model_ckpt: Path,
    model_device: str,
    img_size: int,
    no_trace: bool,
    inference_augment: bool,
    conf_thres: float,
    iou_thres: float,
    agnostic_nms: bool,
    save_dir: Optional[Path],
    save_top_k: Optional[int],
    save_vid: bool,
) -> None:
    """
    Updated version of object detection script for use in generating object
    detection results based on an input COCO file's video/image specifications.

    Expected use-case: generate object detections for video frames (images)
    that we have activity classification truth for.

    \b
    Example:
        python3 python-tpl/yolov7/yolov7/detect_ptg.py \\
            -i ~/data/darpa-ptg/tcn_training_example/train-activity_truth.coco.json \\
            -o ~/data/darpa-ptg/tcn_training_example/train-object_detections.coco.json \\
            --model-hands ./model_files/object_detector/hands_model.pt \\
            --model-objects ./model_files/object_detector/m2_det.pt \\
            --model-device 0 \\
            --no-trace \\
            --img-size 768 \\
            --save-img ./det-debug-imgs \\
            --top-k 4
    """
    # Load the guiding COCO dataset.
    guiding_dset = kwcoco.CocoDataset(input_coco_file, bundle_dpath=img_root)

    # Prevent overwriting an existing file. These are expensive to compute so
    # we don't want to mess that up.
    if output_coco_file.is_file():
        raise ValueError(
            f"Output COCO file already exists, refusing to overwrite: "
            f"{output_coco_file}"
        )
    output_coco_file.parent.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset()
    dset.fpath = output_coco_file.as_posix()

    # Load model
    device, model, stride, imgsz = load_model(model_device, objs_model_ckpt, img_size)
    hand_model = YOLO(hand_model_ckpt)

    if not no_trace:
        model = TracedModel(model, device, img_size)

    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Port over the videos and images sections from the input dataset to the
    # new one.
    dset.dataset['videos'] = guiding_dset.dataset['videos']
    dset.dataset['images'] = guiding_dset.dataset['images']
    dset.index.build(dset)
    # Equality can later be tested with:
    #   guiding_dset.index.videos == dset.index.videos
    #   guiding_dset.index.imgs == dset.index.imgs

    # Add categories
    for i, object_label in enumerate(names):
        dset.ensure_category(name=object_label, id=i)
    # Inject categories for the hand-model additions.
    left_hand_cid = dset.ensure_category(name="hand (left)")
    right_hand_cid = dset.ensure_category(name="hand (right)")
    hands_cid_to_cat = {left_hand_cid: "hand (left)",
                        right_hand_cid: "hand (right)"}

    for vid_id in ub.ProgIter(
        dset.videos(),
        desc="Processing videos",
        verbose=3,
    ):
        vid_obj = dset.index.videos[vid_id]  # noqa
        vid_img_ids = dset.index.vidid_to_gids[vid_id]

        save_imgs_dir: Optional[Path] = None
        if save_dir is not None:
            save_imgs_dir = save_dir / Path(vid_obj["name"]).stem
            assert not save_imgs_dir.is_file()
            save_imgs_dir.mkdir(parents=True, exist_ok=True)
        # If video saving is enable, this list should be populated with the
        # filepaths to the image files that compose the frames of that video
        vid_frames: List[str] = []
        if save_vid:
            # prepopulate "slots" for each frame.
            vid_frames = [""] * len(vid_img_ids)

        for img_id in ub.ProgIter(vid_img_ids, desc='Processing images'):
            img_obj = dset.index.imgs[img_id]

            img_path = Path(dset.get_image_fpath(img_id))
            assert img_path.is_file()
            img0 = cv2.imread(img_path.as_posix())
            height, width = img0.shape[:2]
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            object_preds = predict_image(
                img0, device, model, stride, imgsz, half, inference_augment,
                conf_thres, iou_thres, None, agnostic_nms,
            )
            object_boxes, object_confs, object_class_ids = object_preds

            hands_preds = hand_model.predict(
                source=img0,
                conf=0.1,
                imgsz=img_size,
                device=device,
                verbose=False,
            )[0]  # returns list of length=num images, which is always 1 here.

            top_k_preds = {}
            for xyxy, conf, cls_id in zip(object_boxes, object_confs, object_class_ids):
                # print(f"xyxy 1 : {type(xyxy)}")
                norm_xywh = (xyxy2xywh(xyxy.view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                cxywh = [norm_xywh[0] * width, norm_xywh[1] * height,
                         norm_xywh[2] * width, norm_xywh[3] * height]  # center xy, wh
                xywh = [cxywh[0] - (cxywh[2] / 2), cxywh[1] - (cxywh[3] / 2),
                        cxywh[2], cxywh[3]]

                # print(f"norm_xywh: {norm_xywh}")
                # print(f"cxywh: {cxywh}")
                # print(f"xywh: {xywh}")
                ann = {
                    "area": xywh[2] * xywh[3],
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": xywh,
                    "score": float(conf),
                }
                dset.add_annotation(**ann)

                # Optionally draw results
                if save_dir is not None and save_top_k is None:  # Add bbox to image
                    if cls_id != 0:
                        label = f'{names[int(cls_id)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls_id)], line_thickness=1)

                # Determine top k results to draw later
                if save_top_k is not None:
                    cls_id_index = int(cls_id)
                    k = {'conf': conf, "xyxy": xyxy}

                    if cls_id_index not in top_k_preds.keys():
                        top_k_preds[cls_id_index] = []
                    if len(top_k_preds[cls_id_index]) < save_top_k:
                        top_k_preds[cls_id_index].append(k)
                    else:
                        # Determine if we should add k
                        min_k = min(enumerate(top_k_preds[cls_id_index]), key=lambda x: float(x[1]["conf"]))

                        if float(conf) > float(min_k[1]["conf"]):
                            del top_k_preds[cls_id_index][min_k[0]]
                            top_k_preds[cls_id_index].append(k)

            # print(f"hand boxes: {len(hands_preds.boxes)}") right_hand_cid left_hand_cid
            hand_centers = [center.xywh.tolist()[0][0] for center in hands_preds.boxes][:2]
            hands_label = []
            if len(hand_centers) == 2:
                if hand_centers[0] > hand_centers[1]:
                    hands_label.append(right_hand_cid)
                    hands_label.append(left_hand_cid)
                elif hand_centers[0] <= hand_centers[1]:
                    hands_label.append(left_hand_cid)
                    hands_label.append(right_hand_cid)
            elif len(hand_centers) == 1:
                if hand_centers[0] > width // 2:
                    hands_label.append(right_hand_cid)
                elif hand_centers[0] <= width // 2:
                    hands_label.append(left_hand_cid)

            # print(f"hand boxes: {hands_preds.boxes}")
            # print(f"centers: {hand_centers}")
            for bbox, hand_cid in zip(hands_preds.boxes, hands_label):
                norm_xywh = bbox.xywhn.tolist()[0]
                cxywh = [norm_xywh[0] * width, norm_xywh[1] * height,
                         norm_xywh[2] * width, norm_xywh[3] * height]  # xy, wh

                xywh = [cxywh[0] - (cxywh[2] / 2), cxywh[1] - (cxywh[3] / 2),
                        cxywh[2], cxywh[3]]
                # xywh = [cxywh[0] - (cxywh[2] / 2), cxywh[1] - (cxywh[3] / 2),
                #         cxywh[2], cxywh[3]]
                # cls_id = int(bbox.cls.item())
                # cls_name = names[hand_cid]
                conf = bbox.conf.item()

                # print(f"hand cxywh: {bbox.xywh}")
                # print(f"hand xywh: {xywh}")

                ann = {
                    "area": xywh[2] * xywh[3],
                    "image_id": img_id,
                    "category_id": hand_cid,
                    "bbox": xywh,
                    "score": float(conf),
                }
                # exit()
                dset.add_annotation(**ann)
                if save_dir is not None:
                    label = f"{hands_cid_to_cat[hand_cid]}"
                    # xywh_t = torch.tensor(xywh).view(1,4)
                    xyxy_hand = bbox.xyxy.tolist()[0]
                    # print(f"xywh_t: {xywh_t}")
                    # print(f"xywh: {xywh}")
                    # exit()
                    # print(f"xywh 2: {type(xywh_t)}, {xywh_t.shape}")
                    # xyxy_hand = xywh2xyxy(xywh_t).view(-1).tolist()
                    # xyxy_hand = xywh2xyxy(xywh)
                    # print(f"xyxy_hand: {xyxy_hand}")
                    # print(f"img0: {img0.shape}")
                    # print(f"xyxy: {xyxy}")
                    plot_one_box(xyxy_hand, img0, label=label, color=[0, 0, 0], line_thickness=1)
            # if len(hand_centers) == 1:
            #     import matplotlib.pyplot as plt
            #     image_show = dset.draw_image(gid=img_id)
            #     plt.imshow(image_show)
            #     plt.savefig('myfig.png')
            #     exit()
            # Only draw the top k detections per class
            if save_dir is not None and save_top_k is not None:
                for cls_id, preds in top_k_preds.items():
                    if cls_id != 0:
                        for pred in preds:
                            conf = pred["conf"]
                            xyxy = pred["xyxy"]
                            label = f'{names[int(cls_id)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=colors[int(cls_id)], line_thickness=1)

            if save_dir is not None:
                save_path = (save_imgs_dir / img_path.name).as_posix()
                if not cv2.imwrite(save_path, img0):
                    raise RuntimeError(f"Failed to write debug image: {save_path}")
                if save_vid:
                    vid_frames[img_obj["frame_index"]] = save_path

        if save_vid:
            video_save_path = save_dir / f"{Path(vid_obj['name']).stem}-objects.mp4"
            # video = cv2.VideoWriter(video_save_path, 0, 1, (width,height))

            # print(f"frames: {frames}")
            # for image_path in frames:
            #     video.write(cv2.imread(image_path))

            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                vid_frames,
                fps=vid_obj["framerate"]
            )
            clip.write_videofile(video_save_path.as_posix())
            # cv2.destroyAllWindows()
            # video.release()

            print(f"Saved video to: {video_save_path}")
            # exit()

    print(f"Saving output COCO file... ({output_coco_file})")
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved output COCO file: {output_coco_file}")


if __name__ == '__main__':
    detect_v2()
