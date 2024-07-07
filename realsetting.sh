#!/bin/bash

cfg=realsetting.yaml
exp=exp_realsetting
device=cuda:0

# python train.py --cfg=$cfg --exp=$exp --device=$device

test_resume=exp_realsetting/ckpts/RealSetting240000.pth
sample_folder=exp_realsetting/sample_result/step240000

python test.py --cfg=$cfg --resume=$test_resume --sample_set=../datasets/SPA+ --sample_folder=$sample_folder --device=$device --not_save_result --no_patch

python test.py --cfg=$cfg --resume=$test_resume --sample_set=../datasets/RealSnow --sample_folder=$sample_folder --device=$device --not_save_result

python test.py --cfg=$cfg --resume=$test_resume --sample_set=../datasets/REVIDE --sample_folder=$sample_folder --device=$device --not_save_result
