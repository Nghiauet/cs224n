#!/usr/bin/env bash

# conda create -n cs224n_dfp python=3.8
conda activate cs224n_dfp

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.15
pip install explainaboard_client==0.0.7
