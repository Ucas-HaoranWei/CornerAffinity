import os
import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pprint
from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge
from nnet.py_factory import NetworkFactory
import argparse

class_name = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


image_ext = ['jpg', 'jpeg', 'png']
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def parse_args():
    parser = argparse.ArgumentParser(description="Demo CornerNet")
    parser.add_argument("--demo", dest="demo",
                        help="demo image or image folder",
                        default="./demo/demo_inputs", type=str)
    parser.add_argument("--cfg_file", help="config file",
                        default='CornerNet', type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=-1)
    parser.add_argument("--threshold", dest="threshold", type = float,
                        help="threshold of score",
                        default=0.5)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)

    args = parser.parse_args()
    return args


def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)


def kp_decode(nnet, images, K, ae_threshold=0.5, kernel=3):
    detections = nnet.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel)
    detections = detections.data.cpu().numpy()
    return detections


if __name__ == "__main__":
    args = parse_args()
    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])
    print("system config...")
    pprint.pprint(system_configs.full)

    test_iter = system_configs.max_iter if args.testiter is None \
                                        else args.testiter
    print("loading parameters at iteration: {}".format(test_iter))
    print("building neural network...")
    nnet = NetworkFactory(None)
    print("loading parameters...")
    nnet.load_params(test_iter)
    nnet.cuda()
    nnet.eval_mode()
    threshold = args.threshold
    K             = configs["db"]["top_k"]
    ae_threshold  = configs["db"]["ae_threshold"]
    nms_kernel    = 3

    scales        = configs["db"]["test_scales"]
    weight_exp    = 8
    merge_bbox    = False
    categories    = configs["db"]["categories"]
    nms_threshold = configs["db"]["nms_threshold"]
    max_per_image = configs["db"]["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1,
        "exp_soft_nms": 2
    }["linear_soft_nms"]

    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
    top_bboxes = {}

    if os.path.isdir(args.demo):
        image_names = []
        ls = os.listdir(args.demo)
        for file_name in ls:
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(args.demo, file_name))
    else:
        image_names = [args.demo]

    for image_id, image_name in enumerate(image_names):

        image      = cv2.imread(image_name)

        height, width = image.shape[0:2]

        detections = []

        for scale in scales:
            new_height = int(height * scale)
            new_width = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width = new_width | 127

            images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio = out_width / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, mean, std)

            images[0] = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0] = [int(height * scale), int(width * scale)]
            ratios[0] = [height_ratio, width_ratio]

            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            dets = kp_decode(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
            dets = dets.reshape(2, -1, 8)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets = dets.reshape(1, -1, 8)

            _rescale_dets(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)

        classes = detections[..., -1]
        classes = classes[0]
        detections = detections[0]

        # reject detections with negative scores
        keep_inds = (detections[:, 4] > -1)
        detections = detections[keep_inds]
        classes = classes[keep_inds]

        top_bboxes[image_id] = {}
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
            if merge_bbox:
                soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm,
                               weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
            top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

        scores = np.hstack([
            top_bboxes[image_id][j][:, -1]
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

        if 1:
            image = cv2.imread(image_name)
            # image = image1
            bboxes = {}
            # print(top_bboxes[image_id])
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= threshold)
                cat_name = class_name[j]

                cat_size = cv2.getTextSize(
                    cat_name + '0.0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                color = np.random.random((3,)) * 0.6 + 0.4
                color = color * 255
                color = color.astype(np.int32).tolist()
                for bbox in top_bboxes[image_id][j][keep_inds]:
                    sc = bbox[4]
                    bbox = bbox[0:4].astype(np.int32)
                    txt = '{}{:.1f}'.format(cat_name, sc)
                    if bbox[1] - cat_size[1] - 2 < 0:
                        cv2.rectangle(image,
                                      (bbox[0], bbox[1] + 2),
                                      (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                                      color, -1
                                      )
                        cv2.putText(image, txt,
                                    (bbox[0], bbox[1] + cat_size[1] + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                                    )
                    else:
                        cv2.rectangle(image,
                                      (bbox[0], bbox[1] - cat_size[1] - 2),
                                      (bbox[0] + cat_size[0], bbox[1] - 2),
                                      color, -1
                                      )
                        cv2.putText(image, txt,
                                    (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                                    )
                    cv2.rectangle(image,
                                  (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]),
                                  color, 2
                                  )

            print(image_name)
            cv2.imwrite('./demo/demo_results/' + image_name.split('/')[-1],
                        image)

