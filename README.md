# OCT Denoising and Layer Segmentation using Noise2Void and U-Net

This repository implements a **self-supervised denoising** and **deep-learning segmentation** pipeline for **Optical Coherence Tomography (OCT)** B-scans, combining **Noise2Void** (denoising) and **U-Net** (segmentation).

---

## Repository Overview

```text
oct-denoise-unet/
├── examples/                      # Example input/output visuals
│   ├── input_oct.png              # Raw noisy OCT image
│   ├── denoised_oct.png           # Output from Noise2Void
│   ├── segmentation_mask.png      # U-Net segmentation result
│   ├── MyMasks.gif                # Segmentation animation (optional)
│   └── Reslice of MyMasks.gif     # Alternative view (optional)
│
├── ImageDenoising/                # Noise2Void implementation
│   ├── n2v_simple_main.py
│   ├── n2v_simple_training.py
│   ├── n2v_simple_pred.py
│   └── n2v_simple_models.py
│
├── ImageSegmentation/             # U-Net implementation
│   ├── UnetModel.py
│   ├── preprocessing.py
│   └── train.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Example Results

<table>
  <thead>
    <tr>
      <th align="center">Input (Raw OCT)</th>
      <th align="center">Denoised (Noise2Void)</th>
      <th align="center">Segmented (U-Net)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img src="examples/input_oct.png" alt="input" width="260"></td>
      <td align="center"><img src="examples/denoised_oct.png" alt="denoised" width="260"></td>
      <td align="center"><img src="examples/segmentation_mask.png" alt="segmented" width="260"></td>
    </tr>
  </tbody>
</table>

**Figure.** Left: original OCT B-scan with speckle noise. Middle: Noise2Void denoised reconstruction. Right: U-Net segmentation highlighting layer boundaries.

---

## Getting Started

### Installation
```bash
git clone https://github.com/ajung23/oct-denoise-unet.git
cd oct-denoise-unet
pip install -r requirements.txt
```

### Run Noise2Void (denoising)
```bash
cd ImageDenoising
python n2v_simple_main.py
```

### Run U-Net (segmentation)
```bash
cd ImageSegmentation
python train.py
```

---

## Methodology

| Component  | Framework           | Description                                       |
|------------|---------------------|---------------------------------------------------|
| Noise2Void | TensorFlow / Keras  | Self-supervised denoising on noisy images only    |
| U-Net      | PyTorch             | Encoder–decoder CNN for layer segmentation        |
| Dataset    | OCT B-scans         | Example input: `XZ_area-Stack.tiff`               |
| Metrics    | PSNR, SSIM          | Image quality & structural similarity             |

---

## Research Context

Work derived from the **Boston University Tian Lab (Computational Imaging Systems Lab)**:
- Built ground-truth datasets by annotating OCT lung scans (ImageJ + AnnotatorJ).
- Implemented and tuned **Noise2Void** to suppress speckle noise and improve SNR.
- Designed and fine-tuned a hybrid CNN (U-Net + Dense blocks) for better segmentation.
- Reproducible workflow: preprocessing → denoising → segmentation.

---

## Citation
```bibtex
@misc{jung2025octdenoiseunet,
  author       = {Euijin Jung},
  title        = {OCT Denoising and Layer Segmentation using Noise2Void and U-Net},
  year         = {2025},
  howpublished = {\url{https://github.com/ajung23/oct-denoise-unet}}
}
```

## Contact
Euijin Jung • ajung23@bu.edu • https://www.linkedin.com/in/euijin-jung  
Locations: Orlando, FL / Chicago, IL

## License
MIT License © 2025 Euijin Jung
