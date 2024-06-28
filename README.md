<h1 align="center">
CPPF++: Uncertainty-Aware Sim2Real Object Pose Estimation 
 
 by Vote Aggregation

 (Accepted to TPAMI 2024)
</h1>

<div align="center">
<h3>
<a href="https://qq456cvb.github.io">Yang You</a>, Wenhao He, Jin Liu, Hongkai Xiong, Weiming Wang, <a href="https://www.mvig.org/">Cewu Lu</a>
<br>
<br>
<a href='https://arxiv.org/abs/2211.13398'>
  <img src='https://img.shields.io/badge/Arxiv-PDF-orange?style=flat&logo=arxiv&logoColor=orange' alt='Paper PDF'>
</a>
<a href='https://qq456cvb.github.io/projects/cppf++'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
<br>
</h3>
</div>
 
Object pose estimation constitutes a critical area within the domain of 3D vision. While contemporary state-of-the-art methods that leverage real-world pose annotations have demonstrated commendable performance, the procurement of such real training data incurs substantial costs. This paper focuses on a specific setting wherein only 3D CAD models are utilized as a priori knowledge, devoid of any background or clutter information. We introduce a novel method, CPPF++, designed for sim-to-real pose estimation. This method builds upon the foundational point-pair voting scheme of CPPF, reformulating it through a probabilistic view. To address the challenge posed by vote collision, we propose a novel approach that involves modeling the voting uncertainty by estimating the probabilistic distribution of each point pair within the canonical space. Furthermore, we augment the contextual information provided by each voting unit through the introduction of $N$-point tuples. To enhance the robustness and accuracy of the model, we incorporate several innovative modules, including noisy pair filtering, online alignment optimization, and a tuple feature ensemble. Alongside these methodological advancements, we introduce a new category-level pose estimation dataset, named DiversePose 300.
Empirical evidence demonstrates that our method significantly surpasses previous sim-to-real approaches and achieves comparable or superior performance on novel datasets. 

## Results in the Wild (Casual video captured on IPhone 15, no smoothing)
![teaser](./teaser.gif)

## Update Logs
- 2024/06/27 - Add an example for custom object training, also fix a padding issue in training data. Check `train_custom.ipynb` for more details!
- 2024/06/16 - Uploaded code to reproduce our demo.
- 2024/06/15 - Our paper is accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)!
- 2024/04/10 - Add data processing scripts to convert PhoCAL or Wild6D into REAL275's format.
- 2024/04/01 - pretrained models are under `ckpts`!
- Thanks <a href='https://github.com/dvirginz'>@dvirginz</a> for providing the `Dockerfile` to build `shot.cpp`!
- 2024/03/28 - **Huge improvement** on methods! Check our updated Arxiv paper for more details (refresh your browser cache if not updated). CPPF++ has a **much better performance on many datasets** in the wild, e.g., Wild6D, PhoCAL, DiversPose. See code for more details.
- 2023/09/06 - Major update on methods, check our updated Arxiv paper for more details. Now CPPF++ has a much better performance on both NOCS REAL275 and YCB-Video, **using only synthetic CAD models** for training.

## Code
- v1.0.0: Major improvement in the method!
  - We use both DINO and SHOT features for an ensemble model, which shows greater performance even than some of the state-of-the-art supervised methods!
  - To train the DINO model, first run `dataset.py` to dump the training data (DINO forward will take some time and data cannot be generated online). Then run `train_dino.py`.
  - To train the SHOT model, directly run `train_shot.py`.
  - To test on NOCS REAL275, run `eval.py`. You will need the Mask-RCNN's mask from [SAR-Net](https://github.com/hetolin/SAR-Net).
  - You can also use `Dockerfile` to build `shot.cpp`.
- v0.0.1: Initial release. Support training and evaluation on NOCS dataset. 
  - We follow the same data processing pipeline and dependency setup as [CPPF](https://github.com/qq456cvb/CPPF).
  - This implementation is for the legacy method (arxiv v1).

## Training with your own data
To train with your own data, you need to first preprocess and export DINO and SHOT features into `pkl` (pickle) files for later training (this is to speed up the training process as computing these features are sometimes slow). 

To do so, you need to modify the data path and model names to yours on L192 and L212 of class `ShapeNetDirectDataset` in `dataset.py`, and then use `dump_data` function to dump the data. Also, remove transformations related to `flip2nocs` (it was used in NOCS REAL275 evaluation because ShapeNet objects have a different coordinate frame with NOCS objects). Caveat: we assume all the meshes are diagonally normalized (diagonal of bbox set to 1) and use a predefined range of scales to augment its scale (L165 of `dataset.py`). The final model should be in metric of `meters`. If your model is not diagonal normalized and is already in the real-world metric of `meters`, you may want to delete L233 of `dataset.py`. And if your model is much larger or smaller, you may want to adjust the depth range on L226 of `dataset.py`. The parameter `full_rot` indicates whether to train with full SO(3) random rotation sampling as in DiversePose 300 or just a subset of rotations (very small in-plane rotation, positive elevations) as in NOCS REAL275.

Since our method uses an ensemble from both DINO and SHOT features, after exporting the training data, you will need to run both `train_dino.py` and `train_shot.py` to train two separate models, one for visual clues and the other for geometric clues. Again, make sure the path in class `ShapeNetExportDataset` is consistent, and you may want to delete the lines of `blacklists`.

**We also make a tutorial in train_custom.ipynb**. Open an issue if you have any questions and we are glad to help!



## Evaluation on NOCS REAL275 with pretrained checkpoints
Please run `eval.py` directly.
