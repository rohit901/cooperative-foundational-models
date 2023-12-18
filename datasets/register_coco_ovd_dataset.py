import os
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

coco_meta = _get_coco_instances_meta()

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

CUSTOM_SPLITS_COCO = {
    "coco_ovd_train": ("coco/train2017", "coco/annotations/ovd_instances_train2017_base.json"),
    "coco_ovd_val": ("coco/val2017", "coco/annotations/ovd_instances_val2017_basetarget.json"),
}

for key, (image_root, json_file) in CUSTOM_SPLITS_COCO.items():
    register_coco_instances(
        key,
        coco_meta,
        os.path.join(_root, json_file),
        os.path.join(_root, image_root),
    )