import os
import sys
import warnings
import json

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(proj_path)

script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "params.json")

outputs_dir = os.path.normpath(os.path.join(script_dir, "../../outputs/"))

with open(params_path, "r") as f:
    params = json.load(f)

detectron2_dir = params["detectron2_dir"]
visualize = params["visualize"]
lvis_data_split = params["lvis_data_split"]
class_len_per_prompt = params["class_len_per_prompt"]
cfg_file = params["cfg_file"]
rcnn_weight_dir = params["rcnn_weight_dir"]
sam_checkpoint = params["sam_checkpoint"]
gdino_checkpoint = params["gdino_checkpoint"]
maskrcnn_version = params["maskrcnn_version"]

os.environ['DETECTRON2_DATASETS'] = detectron2_dir

import torch
import detectron2.data.transforms as T

from groundingdino.util.inference import load_model
from load_models import load_fully_supervised_trained_model, load_clip_model, load_sam_model
from utils import get_text_prompt_list_for_g_dino, get_coco_to_lvis_mapping
from evaluation import CustomEvaluator, LVISEvaluatorCustom, inference

from pathlib import Path
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetMapper
from detectron2.evaluation import print_csv_format
from datasets.register_lvis_val_subset import lvis_meta_val_subset # to register the custom lvis_v1_val_subset dataset.
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

Path(outputs_dir).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


model = load_model("cfg/GroundingDINO/GDINO.py", gdino_checkpoint)
model = model.to(device)

rcnn_model, cfg = load_fully_supervised_trained_model(cfg_file, rcnn_weight_dir)

coco_to_lvis = get_coco_to_lvis_mapping(cfg, lvis_data_split)

clip_model, preprocess, text_features, lvis_classes = load_clip_model(lvis_data_split, device)

sam = load_sam_model(device, sam_checkpoint)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

known_class_ids=[3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154, 169, 173, 183,
                         207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445, 469, 474, 496, 534, 569,
                         611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816, 837, 881, 912, 923, 943, 961, 962,
                         964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079, 1095, 1097, 1102, 1112, 1115, 1123, 1133,
                         1139, 1190, 1202]

if maskrcnn_version == "V1":
    test_loader = build_detection_test_loader(
        dataset = get_detection_dataset_dicts(names = lvis_data_split, filter_empty=False),
        mapper= DatasetMapper(
            is_train = False,
            augmentations=[
                T.ResizeShortestEdge(short_edge_length=800, max_size=1333),
            ],
            image_format="RGB", # has to be 'BGR' for MaskRCNN-V2, 'RGB' for MaskRCNN-V1
        ),
        num_workers=4,
    )
elif maskrcnn_version == "V2":
    test_loader = build_detection_test_loader(
        dataset = get_detection_dataset_dicts(names = lvis_data_split, filter_empty=False),
        mapper= DatasetMapper(
            is_train = False,
            augmentations=[
                T.ResizeShortestEdge(short_edge_length=800, max_size=1333),
            ],
            image_format="BGR", # has to be 'BGR' for MaskRCNN-V2, 'RGB' for MaskRCNN-V1
        ),
        num_workers=4,
    )

tokenizer = model.tokenizer

text_prompt_list, positive_map_list = get_text_prompt_list_for_g_dino(lvis_data_split, tokenizer, class_len_per_prompt)

discovery_evaluator = CustomEvaluator(
    evaluator = LVISEvaluatorCustom(
        dataset_name = lvis_data_split,
        distributed = False,
        output_dir = outputs_dir,
        known_class_ids = known_class_ids,    
    ),
)

param_dict = {}
param_dict["visualize"] = visualize
param_dict["out_dir"] = outputs_dir
param_dict["lvis_data_split"] = lvis_data_split
param_dict["class_len_per_prompt"] = class_len_per_prompt
param_dict["positive_map_list"] = positive_map_list
param_dict["rcnn_model"] = rcnn_model

param_dict["clip_model"] = clip_model
param_dict["preprocess"] = preprocess
param_dict["text_features"] = text_features
param_dict["device"] = device

param_dict["coco_to_lvis"] = coco_to_lvis

param_dict["sam"] = sam
param_dict["resize_transform"] = resize_transform
param_dict["maskrcnn_version"] = maskrcnn_version

if __name__ == "__main__":
    results = inference(test_loader, discovery_evaluator, model, text_prompt_list, param_dict)
    print_csv_format(results)

