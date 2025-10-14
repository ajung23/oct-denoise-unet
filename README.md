# OCT Denoising and Layer Segmentation using Noise2Void and U-Net

This repository presents an implementation of a **self-supervised denoising** and **deep-learning-based segmentation** pipeline for **Optical Coherence Tomography (OCT)** B-scans.  
The objective is to improve OCT image quality and delineate structural boundaries using a hybrid workflow combining **Noise2Void** (for denoising) and **U-Net** (for segmentation).

---

## Overview

oct-denoise-unet/
│
├── examples/ # Example input/output visuals
│ ├── input_oct.png # Raw noisy OCT image
│ ├── denoised_oct.png # Output from Noise2Void
│ ├── segmentation_mask.png # U-Net segmentation result
│ ├── MyMasks.gif # Segmentation animation
│ └── Reslice of MyMasks.gif
│
├── ImageDenoising/ # Noise2Void implementation
│ ├── n2v_simple_main.py
│ ├── n2v_simple_training.py
│ ├── n2v_simple_pred.py
│ └── n2v_simple_models.py
│
├── ImageSegmentation/ # U-Net implementation
│ ├── UnetModel.py
│ ├── preprocessing.py
│ └── train.py
│
├── requirements.txt
├── LICENSE
└── README.md

---

## Example Results

| Input (Raw OCT) | Denoised (Noise2Void) | Segmented (U-Net) |
|:--:|:--:|:--:|
| ![](examples/input_oct.png) | ![](examples/denoised_oct.png) | ![](examples/segmentation_mask.png) |

**Figure:** Representative denoising and segmentation outputs.  
Left: Original OCT B-scan with speckle noise.  
Middle: Noise2Void denoised reconstruction.  
Right: U-Net segmentation highlighting major tissue boundaries.

---

## Getting Started

### Installation

```bash
git clone https://github.com/ajung23/oct-denoise-unet.git
cd oct-denoise-unet
pip install -r requirements.txt

# Denoising (Noise2Void)
cd ImageDenoising
python n2v_simple_main.py

# Segmentation (U-Net)
cd ImageSegmentation
python train.py


# Methodology
| Component      | Framework          | Description                                                |
| -------------- | ------------------ | ---------------------------------------------------------- |
| **Noise2Void** | TensorFlow / Keras | Self-supervised denoising trained directly on noisy images |
| **U-Net**      | PyTorch            | Encoder-decoder architecture for layer segmentation        |
| **Dataset**    | OCT B-scans        | Input: `XZ_area-Stack.tiff`                                |
| **Evaluation** | PSNR, SSIM         | Quantitative image quality assessment                      |

# Research Context

This work explores the intersection of self-supervised denoising and supervised segmentation within medical imaging pipelines.
It is particularly relevant to:

Researchers studying speckle noise reduction in OCT or ultrasound imaging

Labs focused on retinal or dermatologic layer analysis

Developers seeking reproducible hybrid TensorFlow–PyTorch workflows

# Citation
@misc{jung2025octdenoiseunet,
  author       = {Euijin Jung},
  title        = {OCT Denoising and Layer Segmentation using Noise2Void and U-Net},
  year         = {2025},
  howpublished = {\url{https://github.com/ajung23/oct-denoise-unet}}
}

# Contact

Euijin Jung
Email: ajung23@bu.edu

LinkedIn: linkedin.com/in/euijin-jung-5378b6203

Locations: Boston, MA / Chicago, IL
