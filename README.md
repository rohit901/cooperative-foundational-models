# Enhancing Novel Object Detection via Cooperative Foundational Models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhancing-novel-object-detection-via/open-vocabulary-object-detection-on-mscoco)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=enhancing-novel-object-detection-via) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhancing-novel-object-detection-via/novel-object-detection-on-lvis-v1-0-val)](https://paperswithcode.com/sota/novel-object-detection-on-lvis-v1-0-val?p=enhancing-novel-object-detection-via)

[Rohit K Bharadwaj](https://rohit901.github.io), [Muzammal Naseer](https://muzammal-naseer.com/), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Khan](https://sites.google.com/view/fahadkhans/home)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.12068)

Official code for our paper "Enhancing Novel Object Detection via Cooperative Foundational Models"

## :rocket: News
* **(Dec 19, 2023)**
  * Code for our method on novel object detection, and open-vocabulary detection setting has been released.

<hr>


![method-diagram](https://rohit901.github.io/media/cooperative_foundational_models/architecture.png)
> **Abstract:** *In this work, we address the challenging and emergent problem of novel object detection (NOD), focusing on the accurate detection of both known and novel object categories during inference. Traditional object detection algorithms are inherently closed-set, limiting their capability to handle NOD. We present a novel approach to transform existing closed-set detectors into open-set detectors. This transformation is achieved by leveraging the complementary strengths of pre-trained foundational models, specifically CLIP and SAM, through our cooperative mechanism. Furthermore, by integrating this mechanism with state-of-the-art open-set detectors such as GDINO, we establish new benchmarks in object detection performance. Our method achieves 17.42 mAP in novel object detection and 42.08 mAP for known objects on the challenging LVIS dataset. Adapting our approach to the COCO OVD split, we surpass the current state-of-the-art by a margin of 7.2 AP<sub>50</sub> for novel classes.*
>

## :trophy: Achievements and Features

- We establish **state-of-the-art results (SOTA)** in novel object detection on LVIS, and open-vocabulary detection benchmark on COCO.
- We propose a simple, modular, and training-free approach which can detect (i.e. localize and classify) known as well as novel objects in the given input image.
- Our approach easily transforms any existing closed-set detectors into open-set detectors by leveraging the complimentary strengths of foundational models like CLIP and SAM.
- The modular nature of our approach allows us to easily swap out any specific component, and further combine it with existing SOTA open-set detectors to achieve additional performance improvements.

## :hammer_and_wrench: Setup and Installation
We have used `python=3.8.15`, and `torch=1.10.1` for all the code in this repository. It is recommended to follow the below steps and setup your conda environment in the same way to replicate the results mentioned in this paper and repository.

1. Clone this repository into your local machine as follows:
```bash
git clone git@github.com:rohit901/cooperative-foundational-models.git
```
or
```bash
git clone https://github.com/rohit901/cooperative-foundational-models.git
```
2. Change the current directory to the main project folder (cooperative-foundational-models):
```bash
cd cooperative-foundational-models
```
3. To install the project dependencies and libraries, use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and install the defined environment from the .yml file by running:
```bash
conda env create -f environment.yml
```
4. Activate the newly created conda environment:
```bash
conda activate coop_foundation_models 
```
5. Install the [Detectron2](https://github.com/facebookresearch/detectron2) v0.6 library via pip:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

### Datasets
To download and setup the required datasets used in this work, please follow these steps:
1. Download the COCO2017 dataset from their official website: [https://cocodataset.org/#download](https://cocodataset.org/#download). Specfically, download `2017 Train images`, `2017 Val images`, `2017 Test images`, and their annotation files `2017 Train/Val annotations`.
2. Download the LVIS v1.0 annotations from: [https://www.lvisdataset.org/dataset](https://www.lvisdataset.org/dataset). There is no need to download images from this website as LVIS uses the same COCO2017 images. Specifically download the annotation files corresponding to the training set (1GB), and validation set (192 MB).
3. Download extra/custom annotation files for COCO open-vocabulary splits from: [COCO-OVD-Annotations](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/rohit_bharadwaj_mbzuai_ac_ae/EgiIumWrcqhGpanKRaYfGYsBOga7V4fgDHcz_W_ys8UVLg?e=Qq0gfT), specifically download both `ovd_instances_train2017_base.json`, and `ovd_instances_val2017_basetarget.json`.
4. Download extra/custom annotation file for `lvis_val_subset` dataset from: [LVIS-Val-Subset](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/rohit_bharadwaj_mbzuai_ac_ae/ErCHDEIltBNBpnedS2S0TUsBFHs1SmVTM525z6ukoiMFLw?e=QPBEL0), specifically download `lvis_v1_val_subset.json`.
5. Detectron2 requires you to setup the datasets in a specific folder format/structure, for that it uses the environment variable `DETECTRON2_DATASETS` which is set equal to the path of the location containing all the different datasets. The file structure of `DETECTRON2_DATASETS` should be as follows:
- `coco/`
  - `annotations/`
    - `instances_train2017.json`
    - `instances_val2017.json`
    - `ovd_instances_train2017_base.json`
    - `ovd_instances_val2017_basetarget.json`
    - `..other coco annotation json files (optional)..`
  - `train2017/`
  - `val2017/`
  - `test2017/`
- `lvis/`
  - `lvis_v1_val.json`
  - `lvis_v1_train.json`
  - `lvis_v1_val_subset.json`

 The above file structure can also be seen from this onedrive link: [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/rohit_bharadwaj_mbzuai_ac_ae/Ej02TuxaTthBrD-yKSOd0SYByEwb5Hqq17Kv4V21AOohkQ?e=H7G9I5). Thus, the value for `DETECTRON2_DATASETS` or `detectron2_dir` in our code file should be the absolute path to the `datasets` directory which follows the above structure.

 ### Model Weights
 All the pre-trained model weights can be downloaded from this link: [model weights](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/rohit_bharadwaj_mbzuai_ac_ae/EpiLUqdhaSxKjB_BjVXCrmAB4cEGNbg3ilGppX5FTQ9sFA?e=EyZs0k). The folder contains the following model weights:

 - **GDINO_weights.pth**: Grounding DINO model weights used in both Novel Object Detection, and Open Vocabulary Detection tasks. Edit the key corresponding to `gdino_checkpoint` in `params.json` file to point to this file.
 - **SAM_weights.pth** Segment Anything Model (SAM) weights used in both Novel Object Detection, and Open Vocabulary Detection tasks. Edit the key corresponding to `sam_checkpoint` in `params.json` file to point to this file.
 - **maskrcnn_v2** Folder containing the weights of trained Mask-RCNN V2 model on COCO datset. These weights are used only in Novel Object Detection task. Do not rename the folder name or the model weight inside the folder, edit the key corresponding to `rcnn_weight_dir` in `scripts/novel_object_detection/params.json` file to point to this folder.
 - **moco_v2_800ep_pretrain.pkl** Initial pre-trained checkpoint for Mask-RCNN. These are used when training Mask-RCNN from scratch on the COCO OVD data split, and thus only used in Open Vocabulary Detection task. Edit the value of `CHECKPOINT_PATH` in `scripts/open_vocab_detection/train_mask_rcnn/train.batch` to point to this file.
 - **MaskRCNN_COCO_OVD** Folder containing the weights of trained Mask-RCNN model on COCO OVD data split. These weights are used only to evaluate performance in Open Vocabulary Detection task. Do not rename folder name or model weight inside the folder, edit the key corresponding to `rcnn_weight_dir` in `scripts/open_vocab_detection/evaluate_method/params.json` file to point to this folder.
 

## :mag_right: Novel Object Detection on LVIS Val Dataset

| Method               | Mask-RCNN   | GDINO       | VLM   | Novel AP | Known AP | All AP  |
|----------------------|-------------|-------------|-------|----------|----------|---------|
| K-Means        | -           | -           | -     | 0.20     | 17.77    | 1.55    |
| Weng et al| -           | -           | -     | 0.27     | 17.85    | 1.62    |
| ORCA            | -           | -           | -     | 0.49     | 20.57    | 2.03    |
| UNO             | -           | -           | -     | 0.61     | 21.09    | 2.18    |
| RNCDL           | V1          | -           | -     | 5.42     | 25.00    | 6.92    |
| GDINO           | -           | ✔           | -     | 13.47    | 37.13    | 15.30   |
| **Ours**             | V2          | ✔           | SigLIP| **17.42**| 42.08    | **19.33**|

**Table 1:** Comparison of object detection performance using mAP on the *lvis_val* dataset.

To replicate our results from the above table (i.e. Table 1 from the main paper):
1. **Modify `scripts/novel_object_detection/params.json` file:**
   - Edit the key `detectron2_dir` and set it following instructions in [Datasets](#datasets)


## Contact
Should you have any questions, please create an issue in this repository or contact at rohit.bharadwaj@mbzuai.ac.ae
