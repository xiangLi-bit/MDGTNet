# All-in-One Weather Removal via Multi-Depth Gated Transformer with Gradient Modulation
PyTorch implementation for "All-in-One Weather Removal via Multi-Depth Gated Transformer with Gradient Modulation".

## Abstract

>All-in-one weather removal methods have made impressive progress recently, but their ability to recover finer details from degraded images still needs to be improved, since (1) the difficulty of Convolutional Neural Networks (CNNs) in providing long-distance information interaction or Visual Transformer with simple convolutions in extracting richer local details, makes them unable to effectively utilize similar original texture features in different regions of a degraded image, and (2) under complex weather degradation distributions, their pixel reconstruction loss functions often result in losing high-frequency details in restored images, even when perceptual loss is used. In this paper, we propose a Multi-Depth Gated Transformer Network (MDGTNet) for all-in-one weather removal, with (1) a multi-depth gated module to capture richer background texture details from various weather noises in an input-adaptive manner, (2) self-attentions to reconstruct similar background textures via long-range feature interaction, and (3) a novel Adaptive Smooth $\text{L}_1$ ($\text{ASL}_1$) loss based on gradient modulation to prompt finer detail restoration. Experimental results show that our method achieves superior performance on both synthetic and real-world benchmarks.

## Requirements

```
torch==2.0.1
torchvision==0.13.0
einops==0.7.0
numpy==1.24.3
opencv_python==4.7.0.72
Pillow==10.2.0
PyYAML==6.0.1
tqdm==4.65.0
yacs==0.1.8
```

## Datasets
| Setting   | Weather Types          | Testing Datasets                           | Training Configurations  |
| :---------: | :----------------------: | :----------------------------------: | :---------------------------------------------------: |
| ***Allweather*** | (Rain, RainDrop, Snow) | ([Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), [RainDrop](https://github.com/rui1996/DeRaindrop), [Snow100K](https://sites.google.com/view/yunfuliu/desnownet)) | Using [All-Weather](https://github.com/jeya-maria-jose/TransWeather) as Training dataset                |
| ***Setting 3*** (***RealSetting***) | (Rain, Haze, Snow)     | ([SPA+](https://github.com/zhuyr97/WGWS-Net), [REVIDE](https://github.com/BookerDeWitt/REVIDE_Dataset), [RealSnow](https://github.com/zhuyr97/WGWS-Net))            | Uniformly sampling 160,000 images           |

**Note**: 

- Please organize the training and testing datasets in the following directory structure and place them in the parent directory of MDGTNet.

    ```bash
    datasets 
     ├─Allweather
     │  ├─train
     │  │  ├─gt
     │  │  └─input
     │  └─val
     │     ├─gt
     │     └─input
     ...
     └─SPA+
        └─sampling
           ├─gt
           └─input
    ```

- ***RealSetting*** corresponds to the real benchmark ***Setting 3*** in the paper

## Pretrained models

Pretrained models coming soon...

## Training
1. The training of ***Allweather***:
```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg=allweather.yaml --exp=exp_allweather
```
2. The training of ***Setting 3***:
```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg=realsetting.yaml --exp=exp_realsetting
```

## Testing

1. The testing of ***Allweather***:
```
# Outdoor_Rain
CUDA_VISIBLE_DEVICES=0 python test.py --cfg=allweather.yaml --resume=exp_allweather/ckpts/Allweather240000.pth --sample_set=../datasets/Outdoor_Rain --sample_folder=exp_allweather/sample_result/step240000 [--not_save_result] --calc_in_Y
# RainDrop
CUDA_VISIBLE_DEVICES=0 python test.py --cfg=allweather.yaml --resume=exp_allweather/ckpts/Allweather240000.pth --sample_set=../datasets/RainDrop --sample_folder=exp_allweather/sample_result/step240000 [--not_save_result] --calc_in_Y
# Snow100K-L
CUDA_VISIBLE_DEVICES=0 python test.py --cfg=allweather.yaml --resume=exp_allweather/ckpts/Allweather240000.pth --sample_set=../datasets/Snow100K-L --sample_folder=exp_allweather/sample_result/step240000 [--not_save_result] --calc_in_Y
# Snow100K-S
CUDA_VISIBLE_DEVICES=0 python test.py --cfg=allweather.yaml --resume=exp_allweather/ckpts/Allweather240000.pth --sample_set=../datasets/Snow100K-S --sample_folder=exp_allweather/sample_result/step240000 [--not_save_result] --calc_in_Y
```
2. The testing of ***Setting 3***:
```
# SPA+
CUDA_VISIBLE_DEVICES=0 python test.py --cfg=realsetting.yaml --resume=exp_realsetting/ckpts/RealSetting240000.pth --sample_set=../datasets/SPA+ --sample_folder=exp_realsetting/sample_result/step240000 [--not_save_result] --no_patch
# SPA+
CUDA_VISIBLE_DEVICES=0 python test.py --cfg=realsetting.yaml --resume=exp_realsetting/ckpts/RealSetting240000.pth --sample_set=../datasets/RealSnow --sample_folder=exp_realsetting/sample_result/step240000 [--not_save_result]
# SPA+
CUDA_VISIBLE_DEVICES=0 python test.py --cfg=realsetting.yaml --resume=exp_realsetting/ckpts/RealSetting240000.pth --sample_set=../datasets/REVIDE --sample_folder=exp_realsetting/sample_result/step240000 [--not_save_result]
```

## Contact

If you have any questions, please contact xl@bit.edu.cn

