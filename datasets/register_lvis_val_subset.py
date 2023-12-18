import os

from detectron2.data.datasets.lvis import get_lvis_instances_meta
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_lvis_json

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

json_path_val = os.path.join(_root, "lvis/lvis_v1_val_subset.json")
image_root_val = os.path.join(_root, "coco/")

lvis_meta_val_subset = get_lvis_instances_meta("lvis_v1")

try:
    DatasetCatalog.get("lvis_v1_val_subset")
except:
    DatasetCatalog.register(
        name="lvis_v1_val_subset",
        func=lambda: load_lvis_json(
            json_file=json_path_val,
            image_root=image_root_val,
            dataset_name="lvis_v1_val_subset"
        )
    )
    MetadataCatalog.get("lvis_v1_val_subset").set(
        json_file=json_path_val, image_root=image_root_val, evaluator_type="lvis", **lvis_meta_val_subset
    )
