# Joint Demosaicing and Deghosting of Time-Varying Exposures for Single-Shot HDR Imaging

### [Project page]() | [Paper](https://vclab.kaist.ac.kr/iccv2023/iccv2023-single-shot-hdr-imaging.pdf) | [Supplemental](https://vclab.kaist.ac.kr/iccv2023/iccv2023-single-shot-hdr-imaging.pdf-supp.pdf)

[Jungwoo Kim](), [Min H. Kim](https://vclab.kaist.ac.kr/minhkim/index.html) KAIST

## Tested Environments

- OS: Ubuntu 16.04 / Windows 10
- Graphic card: TITAN RTX / RTX 2060
- Cuda toolkit version: 10.2
- NVIDIA Driver Version: 456.71
- python: 3.7
- torch version: 1.9.0

## Installation

This repository is built in Pytorch 1.9.0 and tested on Ubuntu 16.04 enviornment (Python3.7, CUDA10.2, cuDNN7.6).

Please follow below instructions:
1. Clone our repository
```
git clone git@github.com:KAIST-VCLAB/singshot-hdr-demosaicing.git
cd singshot-hdr-demosaicing
```

2. Make conda enviornment
```
conda create -n pytorch190 python=3.7
conda activate pytorch190
```

3. Install dependencies
```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -r "requirements.txt"
```

## Demo
First download our [pretrained model](https://drive.google.com/file/d/19W4kWG1YngX10CCT-f9rn7TdqIIpunjc/view?usp=sharing) and put best_psnr_mu.pt file in `code/models/best_psnr_mu.pt`.

To run demo with pre-trained models, run below code:
```
cd code
python test.py
```
