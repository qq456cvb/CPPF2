<div align="center">
  
# CPPF++: Uncertainty-Aware Sim2Real Object Pose Estimation by Vote Aggregation

### IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2024

[![arXiv](https://img.shields.io/badge/arXiv-2211.13398-b31b1b.svg)](https://arxiv.org/abs/2211.13398)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green)](https://qq456cvb.github.io/projects/cppf++)
[![TPAMI](https://img.shields.io/badge/IEEE-TPAMI-blue?style=flat&logo=ieee&logoColor=white)](http://dx.doi.org/10.1109/TPAMI.2024.3419038)

[Yang You](https://qq456cvb.github.io), Wenhao He, Jin Liu, Hongkai Xiong, Weiming Wang, [Cewu Lu](https://www.mvig.org/)

</div>
 
## Overview

CPPF++ is a novel method for **category-level 6D object pose estimation** that bridges the sim-to-real gap using only 3D CAD models for training. Building upon the point-pair voting scheme of CPPF, our approach introduces a probabilistic framework that significantly improves robustness and accuracy in real-world scenarios.

### Key Innovations
- **Uncertainty-aware voting**: Models probabilistic distribution of point pairs to handle vote collisions
- **N-point tuple features**: Enhanced contextual information through multi-point feature aggregation  
- **Sim2Real robustness**: Training exclusively on synthetic CAD models, generalizing to real images
- **DiversePose 300**: New challenging dataset for category-level pose estimation

### Performance Highlights
- ✅ **No real pose annotations** required for training
- ✅ **State-of-the-art** sim2real performance on NOCS REAL275
- ✅ **Superior generalization** to novel datasets (Wild6D, PhoCAL, DiversePose)
- ✅ **Real-time inference** capability for practical applications

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Evaluation](#-evaluation) 
- [Training](#-training)
- [Training with Custom Objects](#-training-with-custom-objects)
- [Datasets](#-datasets)
- [Method Overview](#-method-overview)
- [Related Work](#-related-work)
- [Updates & Changelog](#-updates--changelog)
- [Citation](#-citation)

## 🎬 Results in the Wild

**Casual video captured on iPhone 15 - no smoothing, no object mesh required**
![teaser](./teaser.gif)

Check our new object pose benchmark **[PACE](https://github.com/qq456cvb/PACE)** on *ECCV 2024*.

Also, check our work **[RPMArt](https://r-pmart.github.io/)** on *IROS 2024* that uses CPPF++ to get the pose of articulated objects (e.g., microwaves, drawers).

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+
- CMake 3.16+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/qq456cvb/CPPF2.git
   cd CPPF2
   ```

2. **Install Python dependencies**
   ```bash
   pip install torch torchvision torchaudio numpy opencv-python imageio tqdm omegaconf
   # Additional dependencies for the complete pipeline:
   pip install torch-scatter visdom matplotlib pillow fire
   # For optimization (optional):
   pip install lietorch
   ```

3. **Build SHOT descriptor extraction**
   ```bash
   cd src_shot
   mkdir build && cd build
   cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
```

### Quick Demo

Try our method on your own videos to generate pose estimation GIFs:

```bash
    python demo.py --input_video your_video.mp4 --intrinsics_path camera_intrinsics.npy --object_category mug --output_gif result.gif
```

**Required parameters:**
- `--input_video`: Path to input video file (e.g., `.mp4`, `.avi`)
- `--intrinsics_path`: Path to camera intrinsics numpy file (`.npy`)
- `--object_category`: Target object category (e.g., `mug`, `bottle`, `bowl`)

**Optional parameters:**
- `--output_gif`: Output GIF filename (default: `output.gif`)
- `--input_depth_dir`: Directory containing depth files (optional, for better accuracy)
- `--frame_skip`: Process every N-th frame (default: 1, for faster processing use higher values)
- `--fps`: GIF frame rate (default: 10)

**Camera Intrinsics Format:**
The camera intrinsics should be a 3x3 numpy array saved as `.npy` file:
```python
import numpy as np
# Example camera intrinsics matrix
intrinsics = np.array([
    [fx,  0, cx],
    [ 0, fy, cy], 
    [ 0,  0,  1]
])
np.save('camera_intrinsics.npy', intrinsics)
```

Where `fx`, `fy` are focal lengths and `cx`, `cy` are principal point coordinates.

## 📊 Evaluation

### Pre-trained Models
Download our pre-trained checkpoints from the `ckpts/` directory:
- **DINO model**: Visual feature extraction
- **SHOT model**: Geometric feature extraction  
- **Ensemble**: Combined model for best performance

### Evaluate on NOCS REAL275
```bash
python eval.py
```

*Note: You'll need Mask-RCNN masks from [SAR-Net](https://github.com/hetolin/SAR-Net) for evaluation.*

## 🔧 Training

Our method uses an ensemble of DINO (visual) and SHOT (geometric) features for optimal performance.

### 1. Prepare Training Data
```bash
# Extract and dump training features (this may take some time)
python dataset.py
```
Also
```
conda create -f environment.yml
```

### 2. Train Individual Models
```bash
# Train DINO model (visual features)
python train_dino.py

# Train SHOT model (geometric features)  
python train_shot.py
```

### 3. Custom Object Training
We provide a comprehensive tutorial for training on your own objects:
- 📓 **Interactive Tutorial**: `train_custom.ipynb` 

For detailed instructions, see the [Custom Training Section](#-training-with-custom-objects) below.

## 📚 Training with Custom Objects

📓 **Interactive Tutorial**: Check `train_custom.ipynb` for a complete walkthrough with examples!

## 🗂️ Datasets

### Supported Datasets
- **NOCS REAL275**: Category-level pose estimation benchmark
- **YCB-Video**: Instance-level object pose dataset  
- **Wild6D**: In-the-wild pose estimation
- **PhoCAL**: Photorealistic object dataset
- **DiversePose 300**: Our new diverse pose dataset

### Data Processing Scripts
We provide utilities to convert datasets into REAL275 format:
```bash
python utils/wild6d_convert2real275.py
python utils/phocal_convert2real275.py
```

## 🏗️ Method Overview

### Architecture
CPPF++ extends the point-pair voting framework with several key innovations:

1. **Probabilistic Voting**: Instead of deterministic votes, we model the uncertainty of each point pair's contribution
2. **N-point Tuples**: Enhanced feature representation using multiple point relationships  
3. **Ensemble Learning**: Combining DINO (visual) and SHOT (geometric) features
4. **Uncertainty Estimation**: Explicit modeling of pose estimation confidence

## 🔗 Related Work

- **[CPPF](https://github.com/qq456cvb/CPPF)**: Our previous point-pair voting framework
- **[RPMArt](https://r-pmart.github.io/)**: Extension to articulated objects (IROS 2024)
- **[SAR-Net](https://github.com/hetolin/SAR-Net)**: Source for Mask-RCNN masks used in evaluation

## 📈 Updates & Changelog

- **2024/06/27** - 📓 Added custom object training tutorial (`train_custom.ipynb`)
- **2024/06/16** - 🎬 Demo code release for reproducing results  
- **2024/06/15** - 🎉 **Paper accepted to IEEE TPAMI!**
- **2024/04/10** - 🔧 Data processing scripts for Wild6D and PhoCAL datasets
- **2024/04/01** - 💾 Pre-trained models available in `ckpts/`
- **2024/03/28** - 🚀 **Major performance improvements** across all datasets
- **2023/09/06** - 📊 Significant method updates with enhanced NOCS and YCB-Video performance

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report issues or bugs
- 💡 Suggest new features or improvements  
- 📖 Improve documentation
- 🔧 Submit pull requests

## 📧 Contact

For questions about the paper or code, please:
- Open a GitHub issue for technical problems
- Contact [Yang You](mailto:yangyou@stanford.edu) for research discussions

## 📄 Citation

If you find our work helpful, please consider citing:

```bibtex
@ARTICLE{youcppf2,
  author={You, Yang and He, Wenhao and Liu, Jin and Xiong, Hongkai and Wang, Weiming and Lu, Cewu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={CPPF++: Uncertainty-Aware Sim2Real Object Pose Estimation by Vote Aggregation}, 
  year={2024},
  volume={46},
  number={12},
  pages={9239-9254},
  keywords={Pose estimation;Training;Solid modeling;Noise measurement;Uncertainty;Training data;Filtering;Object pose estimation;sim-to-real transfer;point-pair features;dataset creation},
  doi={10.1109/TPAMI.2024.3419038}
}
```

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
