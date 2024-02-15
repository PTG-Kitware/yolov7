import os
import argparse
import random
import torch
import kwcoco
import glob
import cv2
from typing import List, Optional, Tuple, Union
import warnings

import ubelt as ub
import numpy as np
import numpy.typing as npt

from pathlib import Path

from angel_system.data.medical.data_paths import grab_data
from angel_system.data.common.load_data import time_from_name
from angel_system.data.common.load_data import Re_order

from yolov7.models.yolo import Model as YoloModel
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov7.utils.torch_utils import select_device, TracedModel
from yolov7.utils.plots import plot_one_box
from yolov7.utils.datasets import letterbox


def data_loader(tasks, split):
    """Create a list of all videos in the tasks for the given split

    :return: List of absolute paths to video folders
    """
    training_split = {
        split: []
    }

    for task in tasks:
        ( ptg_root,
        task_data_dir,
        task_activity_config_fn,
        task_activity_gt_dir,
        task_ros_bags_dir,
        task_training_split,
        task_obj_dets_dir,
        task_obj_config ) = grab_data(task, "gyges")
        training_split = {key: value + task_training_split[key] for key, value in training_split.items()}

    print("\nTraining split:")
    for split_name, videos in training_split.items():
        print(f"{split_name}: {len(videos)} videos")
        print([os.path.basename(v) for v in videos])
    print("\n")

    videos = training_split[split]
    return videos


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
    for *xyxy, conf, cls_id in reversed(dets):  # center xy, wh
        yield torch.tensor(xyxy), conf, cls_id.int()


def detect(opt):
    """Run the model over a series of images
    """
    save_path = f"{opt.project}/{opt.name}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Load model
    device, model, stride, imgsz = load_model(opt.device, opt.weights, opt.img_size)

    if not opt.no_trace:
        model = TracedModel(model, device, opt.img_size)

    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    dset = kwcoco.CocoDataset()

    # Add categories
    for i, object_label in enumerate(names):
        if object_label == "background":
            continue
        dset.add_category(name=object_label, id=i)

    videos = data_loader(opt.tasks, opt.split)
    for video in videos:
        video_name = os.path.basename(video)

        if "M2" in video:
            video_task = "M2"
        
        video_data = {
            "name": video_name,
            "task": video_task,
        }
        vid = dset.add_video(**video_data)

        if opt.save_img:
            save_imgs_dir = f"{save_path}/images/{video_name}"
            Path(save_imgs_dir).mkdir(parents=True, exist_ok=True)

        images = glob.glob(f"{video}/images/*.png")
        if not images:
            warnings.warn(f"No images found in {video_name}")
        images = Re_order(images, len(images))
        for image_fn in ub.ProgIter(images, desc=f"images in {video_name}"):
            fn = os.path.basename(image_fn)
            img0 = cv2.imread(image_fn)  # BGR
            assert img0 is not None, 'Image Not Found ' + image_fn
            height, width = img0.shape[:2]

            frame_num, time = time_from_name(image_fn)

            image = {
                "file_name": image_fn,
                "video_id": vid,
                "frame_index": frame_num,
                "width": width,
                "height": height,
            }
            img_id = dset.add_image(**image)

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            preds = predict_image(
                img0, device, model, stride, imgsz, half, opt.augment,
                opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms
            )
            top_k_preds = {}
            for xyxy, conf, cls_id in preds:
                norm_xywh = (xyxy2xywh(xyxy.view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                cxywh = [norm_xywh[0] * width, norm_xywh[1] * height,
                         norm_xywh[2] * width, norm_xywh[3] * height]  # center xy, wh
                xywh = [cxywh[0] - (cxywh[2] / 2), cxywh[1] - (cxywh[3] / 2),
                        cxywh[2], cxywh[3]]

                ann = {
                    "area": xywh[2] * xywh[3],
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": xywh,
                    "confidence": float(conf),
                }
                dset.add_annotation(**ann)

                # Optionaly draw results
                if opt.save_img and not opt.top_k:  # Add bbox to image
                    if cls_id != 0:
                        label = f'{names[int(cls_id)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls_id)], line_thickness=1)
                    
                # Determine top k results to draw later
                if opt.top_k:
                    cls_id_index = int(cls_id)
                    k = {'conf': conf, "xyxy": xyxy}
                    
                    if cls_id_index not in top_k_preds.keys():
                        top_k_preds[cls_id_index] = []
                    if len(top_k_preds[cls_id_index]) < opt.top_k:
                        top_k_preds[cls_id_index].append(k)
                    else:
                        # Determine if we should add k
                        min_k = min(enumerate(top_k_preds[cls_id_index]), key=lambda x: float(x[1]["conf"]))

                        if float(conf) > float(min_k[1]["conf"]):
                            del top_k_preds[cls_id_index][min_k[0]]
                            top_k_preds[cls_id_index].append(k)

            # Only draw the top k detections per class
            if opt.top_k and opt.save_img:
                for cls_id, preds in top_k_preds.items():
                    if cls_id != 0:
                        for pred in preds:
                            conf = pred["conf"]
                            xyxy = pred["xyxy"]
                            label = f'{names[int(cls_id)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=colors[int(cls_id)], line_thickness=1)
                        
            if opt.save_img:
                cv2.imwrite(f"{save_imgs_dir}/{fn}", img0)

    # Save
    dset.fpath = f"{save_path}/{opt.name}_{opt.split}_obj_results.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default='coffee',
        help='Dataset(s)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Data split to run on'
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default='yolov7.pt',
        help='model.pt path(s)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=1280,
        help='inference size (pixels)'
    )
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='object confidence threshold'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=2,
        help='draw the top k detections for each class'
    )
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.45,
        help='IOU threshold for NMS'
    )
    parser.add_argument(
        '--device',
        default='',
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument(
        '--agnostic-nms',
        action='store_true',
        help='class-agnostic NMS'
    )
    parser.add_argument(
        '--project',
        default='runs/detect',
        help='save results to project/name'
    )
    parser.add_argument(
        '--name',
        default='exp',
        help='save results to project/name'
    )
    parser.add_argument(
        '--no-trace',
        action='store_true',
        help='don`t trace model'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='augmented inference'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        help='filter by class: --class 0, or --class 0 2 3'
    )
    parser.add_argument(
        '--save-img',
        action='store_true',
        help='save results to *.png'
    )

    opt = parser.parse_args()
    print(opt)

    detect(opt)


if __name__ == '__main__':
    main()
