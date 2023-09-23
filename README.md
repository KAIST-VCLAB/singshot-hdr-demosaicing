# Joint Demosaicing and Deghosting of Time-Varying Exposures for Single-Shot HDR Imaging

This is the official repository of \
**Joint Demosaicing and Deghosting of Time-Varying Exposures for Single-Shot HDR Imaging.** *Jungwoo Kim, Min H. Kim.* ICCV 2023

<img src="static/teaser.PNG" width="40%" height="50%" title="teaser"></img>

### Challenges

The spatially-varying exposure input has low spatial resolution and different motion blur for each exposure level (b).

Our network restores HDR images (c) from [quad Bayer patterned sensor](https://semiconductor.samsung.com/image-sensor/mobile-image-sensor/isocell-gn1/) images (a) in an end-to-end manner.
We jointly solve demosaicing and deghosting problems to achieve a high-quality snapshot HDR image from the quad-Bayer pattern.

Also, we create a dataset of quad Bayer sensor input with varying exposures and colors using the existing HDR video dataset.

### Abstraction

> The quad-Bayer patterned image sensor has made significant improvements in spatial resolution over recent years due to advancements in image sensor technology. This has enabled single-shot high-dynamic-range (HDR) imaging using spatially varying multiple exposures. Popular methods for multi-exposure array sensors involve varying the gain of each exposure, but this does not effectively change the photoelectronic energy in each exposure. Consequently, HDR images produced using gain-based exposure variation may suffer from noise and details being saturated. To address this problem, we intend to use time-varying exposures in quad-Bayer patterned sensors. This approach allows long-exposure pixels to receive more photon energy than short- or middle-exposure pixels, resulting in higher-quality HDR images. However, time-varying exposures are not ideal for dynamic scenes and require an additional deghosting method. To tackle this issue, we propose a single-shot HDR demosaicing method that takes time-varying multiple exposures as input and jointly solves both the demosaicing and deghosting problems. Our method uses a feature-extraction module to handle mosaiced multiple exposures and a multiscale transformer module to register spatial displacements of multiple exposures and colors. We also created a dataset of quad-Bayer sensor input with time-varying exposures and trained our network using this dataset. Results demonstrate that our method outperforms baseline HDR reconstruction methods with both synthetic and real datasets. With our method, we can achieve high-quality HDR images in challenging lighting conditions.

## Results

### Quantitative Results For Synthetic Dataset

<img src="static/quantitative_syn.PNG" width="60%" height="50%" title="quantitative_syn"></img>

### Qualitative Results For Synthetic Dataset

<img src="static/qualitative_syn.PNG" width="90%" height="50%" title="qualitative_syn"></img>

### Qualitative Results For Real-world Dataset

<img src="static/qualitative_real.PNG" width="90%" height="50%" title="qualitative_real"></img>

Quantitative and qualitative results of our model. Our model outperforms baseline methods in terms of color reconstruction and denoising, particularly in the area with strong motion blur.

## Network Architecture

<img src="static/network.PNG" width="100%" height="50%" title="network"></img>

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
