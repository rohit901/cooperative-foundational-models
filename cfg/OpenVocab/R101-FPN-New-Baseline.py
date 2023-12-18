import detectron2.data.transforms as T
from detectron2.config.lazy import LazyCall as L
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from .mask_rcnn_fpn import model
from .data import dataloader

from .optim import SGD as optimizer
from .train import train

from scripts.open_vocab_detection.coco_eval_utils.custom_coco_eval import CustomCOCOEvaluator

from datasets.register_coco_ovd_dataset import coco_meta # to register the OVD datasets

# train from scratch
train.init_checkpoint = ""
train.amp.enabled = True
train.ddp.fp16_compression = True
model.backbone.bottom_up.freeze_at = 0

# SyncBN
# fmt: off
model.backbone.bottom_up.stem.norm = \
    model.backbone.bottom_up.stages.norm = \
    model.backbone.norm = "SyncBN" # replace this with only "BN" if running on single GPU, or use "SyncBN" with multi-GPU.

# Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
model.roi_heads.box_head.conv_norm = \
    model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c,
                                                                       stats_mode="N")
# fmt: on

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

# resize_and_crop_image in:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/utils/input_utils.py#L127  # noqa: E501, B950
image_size = 1024
dataloader.train.mapper.augmentations = [
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
    L(T.RandomFlip)(horizontal=True),
]

# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

# larger batch-size.
dataloader.train.total_batch_size = 64

# use OVD datasets
dataloader.train.dataset.names = "coco_ovd_train"
dataloader.test.dataset.names = "coco_ovd_val"

# custom evaluator
dataloader.evaluator = L(CustomCOCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)

# Equivalent to 100 epochs.
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 180000

lr_frac1 = 0.888889
lr_frac2 = 0.962961

milestone1 = int(train.max_iter * lr_frac1)
milestone2 = int(train.max_iter * lr_frac2)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[milestone1, milestone2],
        num_updates=train.max_iter,
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.067,
)

optimizer.lr = 0.1
optimizer.weight_decay = 4e-5



model.backbone.bottom_up.stages.depth = 101

lr_multiplier.scheduler.num_updates = train.max_iter

# Set bbox localization head to be class-agnostic
model.roi_heads.box_predictor.cls_agnostic_bbox_reg = True

# Set mask head to be class-agnostic
model.roi_heads.mask_head.num_classes = 1