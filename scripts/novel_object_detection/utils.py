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
from detectron2.utils.file_io import PathManager
from PIL import Image

_EXIF_ORIENT = 274  # exif 'Orientation' tag
# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]

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
            positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def get_text_prompt_list_for_g_dino(lvis_data_split, tokenizer, class_len_per_prompt):
    
    lvis_metadata = MetadataCatalog.get(lvis_data_split)
    lvis_classes = lvis_metadata.get("thing_classes")

    lvis_classes = [i.lower() for i in lvis_classes]
    lvis_classes = [s.replace("_", " ") for s in lvis_classes] # replace _ with space

    length = class_len_per_prompt
    lvis_classes_split = [lvis_classes[i:i + length] for i in range(0, len(lvis_classes), length)]
    
    text_prompt_list = []
    positive_map_list = []
    for lvis_classes_subset in lvis_classes_split:
        captions, cat2tokenspan = build_captions_and_token_span(lvis_classes_subset, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in lvis_classes_subset]
        positive_map = create_positive_map_from_span(tokenizer(captions), tokenspanlist) # shape: (num_categories, 256)
        positive_map_list.append(positive_map)

        text_prompt_list.append(captions)

    return text_prompt_list, positive_map_list

def get_coco_to_lvis_mapping(cfg, lvis_data_split):

    # covert coco_meta_data thing class idx to lvis idx
    coco_to_lvis = {}
    not_found_list = []
    found_list = []
    coco_meta_data = MetadataCatalog.get("coco_2017_train")
    lvis_metadata = MetadataCatalog.get(lvis_data_split)
    lvis_classes = lvis_metadata.get("thing_classes")

    for idx, coco_class in enumerate(coco_meta_data.thing_classes):
        if coco_class in lvis_classes:
            lvis_idx = lvis_classes.index(coco_class)
            coco_to_lvis[idx] = lvis_idx
            found_list.append(coco_class)
        else:
            coco_class = "_".join(coco_class.split(" "))
            if coco_class in lvis_classes:
                lvis_idx = lvis_classes.index(coco_class)
                coco_to_lvis[idx] = lvis_idx
                found_list.append(coco_class)
            else:
                not_found_list.append(coco_class)
                coco_to_lvis[idx] = -1
        

    name_to_idx = {}
    for name in not_found_list:
        name = name.replace("_", " ")
        name_to_idx[name] = coco_meta_data.thing_classes.index(name)


    # initially idx 52 could not be mapped, i.e. "hot dog" in coco
    # mapped idx 52 to 168, i.e. "bun" in lvis
    coco_to_lvis[2] = 206
    coco_to_lvis[5] = 172
    coco_to_lvis[6] = 1114
    coco_to_lvis[10] = 444

    coco_to_lvis[27] = 715
    coco_to_lvis[30] = 963
    coco_to_lvis[32] = 40

    coco_to_lvis[40] =  1189
    coco_to_lvis[49] = 734
    coco_to_lvis[52] = 168
    coco_to_lvis[54] = 386
    coco_to_lvis[57] = 981
    coco_to_lvis[58] = 836
    coco_to_lvis[62] = 1076
    coco_to_lvis[63] = 630
    coco_to_lvis[64] = 704
    coco_to_lvis[65] = 880
    coco_to_lvis[66] = 295

    coco_to_lvis[67] = 229
    coco_to_lvis[68] = 686
    coco_to_lvis[78] = 533

    return coco_to_lvis

def get_clip_preds(img, clip_model, text_features):

    with torch.no_grad(), torch.cuda.amp.autocast():
        img_features = clip_model.encode_image(img) 
        img_features = F.normalize(img_features, dim=-1)

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

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image

def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)