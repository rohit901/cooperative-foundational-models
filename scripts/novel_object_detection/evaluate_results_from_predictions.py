import logging
import argparse
import itertools
import copy
import torch
import os
import sys
import numpy as np

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(proj_path)

os.environ['DETECTRON2_DATASETS'] = "path/to/datasets"

from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from collections import OrderedDict
from detectron2.utils.logger import log_every_n_seconds, create_small_table
from lvis import LVISEval
from datasets.register_lvis_val_subset import lvis_meta_val_subset


def tasks_from_predictions(predictions):
    for pred in predictions:
        if "segmentation" in pred:
            return ("bbox", "segm")
    return ("bbox",)

def _evaluate_predictions_on_lvis(
        logger, lvis_gt, lvis_results, iou_type, max_dets_per_image=None, class_names=None, known_class_ids=None
):
    """
    Same as the original implementation, except that extra evaluation on only known or only novel classes is performed
    if `known_class_ids` is provided. For that replaces object of `LVISEval` with `LVISEvalCustom`.
    """

    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 300  # Default for LVIS dataset

    from lvis import LVISEval, LVISResults

    logger.info(f"[Evaluator new] Evaluating with max detections per image = {max_dets_per_image}")
    lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
    if known_class_ids is not None:
        lvis_eval = LVISEvalCustom(lvis_gt, lvis_results, iou_type, known_class_ids)
    else:
        lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info("[Evaluator new] Evaluation results for {}: \n".format(iou_type) + create_small_table(results))

    if known_class_ids is not None:  # Print results for known and novel classes separately
        for results, subtitle in [
            (lvis_eval.results_known, "known classes only"),
            (lvis_eval.results_novel, "novel classes only"),
        ]:
            results = {metric: float(results[metric] * 100) for metric in metrics}
            logger.info("Evaluation results for {} ({}): \n".format(iou_type, subtitle) + create_small_table(results))

    return results

class LVISEvalCustom(LVISEval):
    """
    Extends `LVISEval` with printing results for known and novel classes only when `known_class_ids` is provided.
    """

    def __init__(self, lvis_gt, lvis_dt, iou_type="segm", known_class_ids=None):
        super().__init__(lvis_gt, lvis_dt, iou_type)

        # Remap categories list following the mapping applied to train data, - that is list all categories in a
        # consecutive order and use their indices; see: `lvis-api/lvis/eval.py` line 109:
        # https://github.com/lvis-dataset/lvis-api/blob/35f09cd7c5f313a9bf27b329ca80effe2b0c8a93/lvis/eval.py#L109
        if known_class_ids is None:
            self.known_class_ids = None
        else:
            self.known_class_ids = [self.params.cat_ids.index(c) for c in known_class_ids]

    def _summarize(
            self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None, subset_class_ids=None
    ):
        """Extends the default version by supporting calculating the results only for the subset of classes."""

        if subset_class_ids is None:  # Use all classes
            subset_class_ids = list(range(len(self.params.cat_ids)))

        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                subset_class_ids = list(set(subset_class_ids).intersection(self.freq_groups[freq_group_idx]))
                s = s[:, :, subset_class_ids, aidx]
            else:
                s = s[:, :, subset_class_ids, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, subset_class_ids, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Extends the default version by supporting calculating the results only for the subset of classes."""

        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        if self.known_class_ids is None:
            eval_groups = [(self.results, None)]
        else:
            cat_ids_mapped_list = list(range(len(self.params.cat_ids)))
            novel_class_ids = list(set(cat_ids_mapped_list).difference(self.known_class_ids))
            self.results_known = OrderedDict()
            self.results_novel = OrderedDict()
            eval_groups = [
                (self.results, None),
                (self.results_known, self.known_class_ids),
                (self.results_novel, novel_class_ids),
            ]

        max_dets = self.params.max_dets

        for container, subset_class_ids in eval_groups:
            container["AP"]   = self._summarize('ap', subset_class_ids=subset_class_ids)
            container["AP50"] = self._summarize('ap', iou_thr=0.50, subset_class_ids=subset_class_ids)
            container["AP75"] = self._summarize('ap', iou_thr=0.75, subset_class_ids=subset_class_ids)
            container["APs"]  = self._summarize('ap', area_rng="small", subset_class_ids=subset_class_ids)
            container["APm"]  = self._summarize('ap', area_rng="medium", subset_class_ids=subset_class_ids)
            container["APl"]  = self._summarize('ap', area_rng="large", subset_class_ids=subset_class_ids)
            container["APr"]  = self._summarize('ap', freq_group_idx=0, subset_class_ids=subset_class_ids)
            container["APc"]  = self._summarize('ap', freq_group_idx=1, subset_class_ids=subset_class_ids)
            container["APf"]  = self._summarize('ap', freq_group_idx=2, subset_class_ids=subset_class_ids)

            key = "AR@{}".format(max_dets)
            container[key] = self._summarize('ar', subset_class_ids=subset_class_ids)

            for area_rng in ["small", "medium", "large"]:
                key = "AR{}@{}".format(area_rng[0], max_dets)
                container[key] = self._summarize('ar', area_rng=area_rng, subset_class_ids=subset_class_ids)

def eval_predictions(predictions, lvis_data_split, known_class_ids):
    """
    Same as `LVISEvaluator`, code had to be re-copied to fix a reference to a new `_evaluate_predictions_on_lvis()`
    that is re-defined below.
    """
    from lvis import LVIS
    lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
    tasks = tasks_from_predictions(lvis_results)

    metadata = MetadataCatalog.get(lvis_data_split)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    json_file = PathManager.get_local_path(metadata.json_file)
    lvis_api = LVIS(json_file)

    results = OrderedDict()

    # LVIS evaluator can be used to evaluate results for COCO dataset categories.
    # In this case `_metadata` variable will have a field with COCO-specific category mapping.
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {
            v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }
        for result in lvis_results:
            result["category_id"] = reverse_id_mapping[result["category_id"]]
    else:
        # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
        for result in lvis_results:
            result["category_id"] += 1

    logger.info("[Evaluator new] Evaluating predictions ...")
    for task in sorted(tasks):
        res = _evaluate_predictions_on_lvis(
            logger,
            lvis_api,
            lvis_results,
            task,
            max_dets_per_image=None,
            class_names=metadata.get("thing_classes"),
            known_class_ids=known_class_ids,
        )
        results[task] = res

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--lvis-data-split", type=str, default="lvis_v1_val")

    args = parser.parse_args()

    predictions = torch.load(args.predictions)
    print("Length of predictions: ", len(predictions))
    data_split = args.lvis_data_split

    known_class_ids=[3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154, 169, 173, 183,
                         207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445, 469, 474, 496, 534, 569,
                         611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816, 837, 881, 912, 923, 943, 961, 962,
                         964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079, 1095, 1097, 1102, 1112, 1115, 1123, 1133,
                         1139, 1190, 1202]

    results = eval_predictions(predictions, data_split, known_class_ids)
