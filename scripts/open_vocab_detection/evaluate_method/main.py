import os
import sys
import warnings
import json

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(proj_path)

script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "params.json")

outputs_dir = os.path.normpath(os.path.join(script_dir, "../../../outputs/coco_ovd/coop_found_models/"))

with open(params_path, "r") as f:
    params = json.load(f)

detectron2_dir = params["detectron2_dir"]
visualize = params["visualize"]
data_split = params["data_split"]
cfg_file = params["cfg_file"]
rcnn_weight_dir = params["rcnn_weight_dir"]
sam_checkpoint = params["sam_checkpoint"]
gdino_checkpoint = params["gdino_checkpoint"]

os.environ['DETECTRON2_DATASETS'] = detectron2_dir

import torch
import detectron2.data.transforms as T

from groundingdino.util.inference import load_model
from load_models import load_fully_supervised_trained_model, load_clip_model, load_sam_model
from utils import get_text_prompt_for_g_dino, get_ovd_id_to_coco_id
from evaluator_loop import inference

from pathlib import Path
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetMapper
from detectron2.evaluation import print_csv_format
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm

from datasets.register_coco_ovd_dataset import coco_meta # to register the OVD datasets
from scripts.open_vocab_detection.coco_eval_utils.custom_coco_eval import CustomCOCOEvaluator

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

Path(outputs_dir).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


model = load_model("cfg/GroundingDINO/GDINO.py", gdino_checkpoint)
model = model.to(device)

rcnn_model, cfg = load_fully_supervised_trained_model(cfg_file, rcnn_weight_dir)

ovd_id_to_coco_id = get_ovd_id_to_coco_id()

clip_model, preprocess, text_features = load_clip_model(device)

sam = load_sam_model(device, sam_checkpoint)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

test_loader = build_detection_test_loader(
    dataset = get_detection_dataset_dicts(names = data_split, filter_empty=False),
    mapper= DatasetMapper(
        is_train = False,
        augmentations=[
            T.ResizeShortestEdge(short_edge_length=800, max_size=1333),
        ],
        image_format="BGR",
    ),
    num_workers=4,
)

tokenizer = model.tokenizer

text_prompt, positive_map = get_text_prompt_for_g_dino(tokenizer)

coco_evaluator = CustomCOCOEvaluator(dataset_name = data_split)

param_dict = {}
param_dict["visualize"] = visualize
param_dict["out_dir"] = outputs_dir
param_dict["data_split"] = data_split
param_dict["positive_map"] = positive_map
param_dict["rcnn_model"] = rcnn_model

param_dict["clip_model"] = clip_model
param_dict["preprocess"] = preprocess
param_dict["text_features"] = text_features
param_dict["device"] = device

param_dict["ovd_id_to_coco_id"] = ovd_id_to_coco_id

param_dict["sam"] = sam
param_dict["resize_transform"] = resize_transform

if __name__ == "__main__":
    results = inference(test_loader, coco_evaluator, model, text_prompt, param_dict)
    print_csv_format(results)

