# OCT Denoising and Layer Segmentation using Noise2Void and U-Net

This repository presents an implementation of a self-supervised denoising and deep-learning-based segmentation pipeline for **Optical Coherence Tomography (OCT)** B-scans.
The goal is to improve OCT image quality and delineate structural tissue boundaries using a hybrid approach that combines **Noise2Void** (denoising) and **U-Net** (segmentation).

---

## Repository Overview

oct-denoise-unet/
|
├── examples/ # Example input/output visuals
| ├── input_oct.png # Raw noisy OCT image
| ├── denoised_oct.png # Output from Noise2Void
| ├── segmentation_mask.png # U-Net segmentation result
| ├── MyMasks.gif # Segmentation animation (optional)
| └── Reslice of MyMasks.gif # Alternative view (optional)
|
├── ImageDenoising/ # Noise2Void implementation
| ├── n2v_simple_main.py
| ├── n2v_simple_training.py
| ├── n2v_simple_pred.py
| └── n2v_simple_models.py
|
├── ImageSegmentation/ # U-Net implementation
| ├── UnetModel.py
| ├── preprocessing.py
| └── train.py
|
├── requirements.txt
├── LICENSE
└── README.md

yaml
코드 복사

---

## Example Results

| Input (Raw OCT) | Denoised (Noise2Void) | Segmented (U-Net) |
|:--:|:--:|:--:|
| ![](examples/input_oct.png) | ![](examples/denoised_oct.png) | ![](examples/segmentation_mask.png) |

**Figure:** Representative denoising and segmentation outputs.  
Left → Original OCT B-scan with speckle noise.  
Middle → Noise2Void denoised reconstruction.  
Right → U-Net segmentation highlighting major tissue boundaries.

---

## Getting Started

### Installation
```bash
git clone https://github.com/ajung23/oct-denoise-unet.git
cd oct-denoise-unet
pip install -r requirements.txt
Run Noise2Void Denoising
bash
코드 복사
cd ImageDenoising
python n2v_simple_main.py
Run U-Net Segmentation
bash
코드 복사
cd ImageSegmentation
python train.py
Methodology
Component	Framework	Description
Noise2Void	TensorFlow / Keras	Self-supervised denoising trained on noisy images only
U-Net	PyTorch	Encoder–decoder CNN for layer/tissue segmentation
Dataset	OCT B-scans	Example input: XZ_area-Stack.tiff
Metrics	PSNR, SSIM	Image quality & structural similarity

This pipeline enables end-to-end OCT enhancement without clean ground-truth, demonstrating the synergy of self-supervised denoising and supervised segmentation.

Research Context
This work stems from the Boston University Tian Lab (Computational Imaging Systems Lab).

Contributions:

Built ground-truth datasets by annotating OCT lung scans (ImageJ + AnnotatorJ).

Implemented and tuned Noise2Void to suppress speckle noise and boost SNR.

Designed and fine-tuned a hybrid CNN (U-Net + Dense blocks) for stronger segmentation.

Established a reproducible workflow: preprocessing → denoising → segmentation.

Citation
bibtex
코드 복사
@misc{jung2025octdenoiseunet,
  author       = {Euijin Jung},
  title        = {OCT Denoising and Layer Segmentation using Noise2Void and U-Net},
  year         = {2025},
  howpublished = {\url{https://github.com/ajung23/oct-denoise-unet}}
}
Contact
Euijin Jung
Email: ajung23@bu.edu
LinkedIn: https://www.linkedin.com/in/euijin-jung
Locations: Orlando, FL / Chicago, IL
