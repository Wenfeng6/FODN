![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.10](https://img.shields.io/badge/PyTorch->=1.10-blue.svg)

### Prepare Datasets

```bash
mkdir data
```
Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset), and the [Occluded_REID](https://github.com/wangguanan/light-reid/blob/master/reid_datasets.md), 
Then unzip them and rename them under the directory like

```
data
├── Occluded_Duke
│   └── images ..
├── Occluded_REID
│   └── images ..
├── market1501
│   └── images ..
└── dukemtmcreid
    └── images ..
```

### Installation

```bash
pip install -r requirements.txt
```

### Prepare ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

## Training

We utilize 1 3090 GPU for training and it takes around 14GB GPU memory.

You can train the FODN with:

```bash
python train.py --config_file configs/fodn.yml MODEL.DEVICE_ID "('your device id')"
```
**Some examples:**
```bash
# Occluded_Duke
python train.py --config_file configs/OCC_Duke/fodn.yml MODEL.DEVICE_ID "('0')"
```

We have set the validation set as Occluded REID when training on the Market-1501. Therefore, if you want to use the Market-1501, please modify it in the 'datasets/market1501.py'.


## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

**Some examples:**
```bash
# OCC_Duke
python test.py --config_file configs/OCC_Duke/fodn.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/occ_duke_dpm/transformer_150.pth'
```

