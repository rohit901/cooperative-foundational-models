import numpy as np
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
import os
import matplotlib as mpl
import random

from detectron2.data import MetadataCatalog
from torch import nn
from detectron2.utils.visualizer import _create_text_labels, Visualizer
from typing import List

from detectron2.utils.visualizer import _create_text_labels, Visualizer
from scripts.open_vocab_detection.coco_eval_utils.coco_ovd_split import categories_seen, categories_unseen

class BBoxVisualizer(Visualizer):
    colors = list(mcolors.BASE_COLORS.keys())
    
    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): an :class:`Instances` object with fields "pred_boxes", "pred_classes", and "scores"

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes.tensor
        scores = predictions.scores
        classes = predictions.pred_classes
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            color = self.colors[idx % len(self.colors)]
            box = box.cpu().detach().numpy()
            x0, y0, x1, y1 = box
            self.draw_box((x0, y0, x1, y1), edge_color = color, linewidth = 4.0)
            # Draw label at the top-left corner of the bounding box
            self.draw_text(label, (x0, y0), horizontal_alignment="left", color = color, font_size = 12)
        return self.output

    def draw_box(self, box_coord, linewidth, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

def build_captions_and_token_span(cat_list, force_lowercase):
    """
    Return:
        captions: str
        cat2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """

    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list: List = class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name: str = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan

def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def get_ovd_id_to_coco_id():
    seen_names = [x['name'] for x in categories_seen]
    unseen_names = [x['name'] for x in categories_unseen]

    coco_ovd_classes = seen_names + unseen_names

    all_coco_classes = MetadataCatalog.get("coco_2017_val").get("thing_classes")
    ovd_id_to_coco_id = {}

    for i, coco_class in enumerate(all_coco_classes):
        if coco_class in coco_ovd_classes:
            ovd_id_to_coco_id[coco_ovd_classes.index(coco_class)] = i

    return ovd_id_to_coco_id

def get_text_prompt_for_g_dino(tokenizer):
    seen_names = [x['name'] for x in categories_seen]
    unseen_names = [x['name'] for x in categories_unseen]

    coco_ovd_classes = seen_names + unseen_names

    coco_ovd_classes = [i.lower() for i in coco_ovd_classes]
    coco_ovd_classes = [s.replace("_", " ") for s in coco_ovd_classes] # replace _ with space

    captions, cat2tokenspan = build_captions_and_token_span(coco_ovd_classes, True)
    tokenspanlist = [cat2tokenspan[cat] for cat in coco_ovd_classes]
    positive_map = create_positive_map_from_span(tokenizer(captions), tokenspanlist) # shape: (num_categories, 256)

    return captions, positive_map

def get_clip_preds(img, clip_model, text_features):
    """
    img: torch.Size([N, 3, 224, 224])
    text_features: torch.Size([768, 1203])
    """
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_features = clip_model.encode_image(img) # features shape: torch.Size([50, N, 768])
        img_features = F.normalize(img_features, dim=-1) # features shape: torch.Size([50, N, 768])

        text_probs = torch.sigmoid(img_features @ text_features.T * clip_model.logit_scale.exp() + clip_model.logit_bias) # shape: torch.Size([N, 1203])

        values, indices = text_probs.topk(1)
    
    return values, indices

def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res