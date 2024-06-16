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
<a href='#'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
<a href='#'>
<img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Video'/>
</a>
<br>
</h3>
</div>
 
Object pose estimation constitutes a critical area within the domain of 3D vision. While contemporary state-of-the-art methods that leverage real-world pose annotations have demonstrated commendable performance, the procurement of such real training data incurs substantial costs. This paper focuses on a specific setting wherein only 3D CAD models are utilized as a priori knowledge, devoid of any background or clutter information. We introduce a novel method, CPPF++, designed for sim-to-real pose estimation. This method builds upon the foundational point-pair voting scheme of CPPF, reformulating it through a probabilistic view. To address the challenge posed by vote collision, we propose a novel approach that involves modeling the voting uncertainty by estimating the probabilistic distribution of each point pair within the canonical space. Furthermore, we augment the contextual information provided by each voting unit through the introduction of $N$-point tuples. To enhance the robustness and accuracy of the model, we incorporate several innovative modules, including noisy pair filtering, online alignment optimization, and a tuple feature ensemble. Alongside these methodological advancements, we introduce a new category-level pose estimation dataset, named DiversePose 300.
Empirical evidence demonstrates that our method significantly surpasses previous sim-to-real approaches and achieves comparable or superior performance on novel datasets. 

## Results in the Wild (Casual video captured on IPhone 15, no smoothing)
![teaser](./teaser.gif)

## Update Logs
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
