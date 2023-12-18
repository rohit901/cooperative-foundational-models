import open_clip
import torch
import pickle

from detectron2.data import MetadataCatalog

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry
from utils import article, processed_name

def load_fully_supervised_trained_model(cfg_file, weight_dir):
    # Load the model weights of supevised training phase
    opts = [f'train.output_dir={weight_dir}', f'train.init_checkpoint={weight_dir}/model_final.pth']

    cfg = LazyConfig.load(cfg_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    default_setup(cfg, None)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)

    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    return model, cfg

def load_clip_model(data_split, device):
    # Load the SigLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')

    clip_model = clip_model.to(device)


    lvis_metadata = MetadataCatalog.get(data_split)
    lvis_classes = lvis_metadata.get("thing_classes")

    with open('lvis_original_class_to_synonyms.pkl', 'rb') as f:
        class_names_to_synonyms = pickle.load(f)

    templates = [
        "There is {article} {} in the scene.",
        "There is the {} in the scene.",
        "a photo of {article} {} in the scene.",
        "a photo of the {} in the scene.",
        "a photo of one {} in the scene.",
        "itap of {article} {}.",
        "itap of my {}.",
        "itap of the {}.",
        "a photo of {article} {}.",
        "a photo of my {}.",
        "a photo of the {}.",
        "a photo of one {}.",
        "a photo of many {}.",
        "a good photo of {article} {}.",
        "a good photo of the {}.",
        "a bad photo of {article} {}.",
        "a bad photo of the {}.",
        "a photo of a nice {}.",
        "a photo of the nice {}.",
        "a photo of a cool {}.",
        "a photo of the cool {}.",
        "a photo of a weird {}.",
        "a photo of the weird {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of a clean {}.",
        "a photo of the clean {}.",
        "a photo of a dirty {}.",
        "a photo of the dirty {}.",
        "a bright photo of {article} {}.",
        "a bright photo of the {}.",
        "a dark photo of {article} {}.",
        "a dark photo of the {}.",
        "a photo of a hard to see {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of {article} {}.",
        "a low resolution photo of the {}.",
        "a cropped photo of {article} {}.",
        "a cropped photo of the {}.",
        "a close-up photo of {article} {}.",
        "a close-up photo of the {}.",
        "a jpeg corrupted photo of {article} {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of {article} {}.",
        "a blurry photo of the {}.",
        "a pixelated photo of {article} {}.",
        "a pixelated photo of the {}.",
        "a black and white photo of the {}.",
        "a black and white photo of {article} {}.",
        "a plastic {}.",
        "the plastic {}.",
        "a toy {}.",
        "the toy {}.",
        "a plushie {}.",
        "the plushie {}.",
        "a cartoon {}.",
        "the cartoon {}.",
        "an embroidered {}.",
        "the embroidered {}.",
        "a painting of the {}.",
        "a painting of a {}."
    ]

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = []
        for classname in lvis_classes:
            syn_features = []
            for syn in class_names_to_synonyms[classname]:
                texts = [template.format(processed_name(syn, rm_dot=True),
                        article=article(syn)) for template in templates]
                texts = [
                    'This is ' + text if text.startswith('a') or text.startswith('the') else text
                    for text in texts
                ]
                texts = tokenizer(texts, context_length = clip_model.context_length).to(device)
                syn_embeddings = clip_model.encode_text(texts)
                syn_embeddings /= syn_embeddings.norm(dim=-1, keepdim=True)
                syn_embedding = syn_embeddings.mean(dim=0)
                syn_embedding /= syn_embedding.norm()
                syn_features.append(syn_embedding)

            syn_features = torch.stack(syn_features, dim=0).to(device)
            syn_feature = syn_features.mean(dim=0)
            syn_feature /= syn_feature.norm() # shape: (1152)
            text_features.append(syn_feature)

        text_features = torch.stack(text_features, dim=0).to(device) # shape: (1203, 1152)
    
    return clip_model, preprocess, text_features, lvis_classes

def load_sam_model(device, sam_checkpoint):
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return sam