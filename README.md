<h1 align="center">
Go Beyond Point Pairs: 

A General and Accurate Sim2Real Object Pose Voting Method

 with Efficient Online Synthetic Training
</h1>

<div align="center">
<h3>
<a href="https://qq456cvb.github.io">Yang You</a>, Wenhao He, Michael Xu LIU, Weiming Wang, <a href="https://www.mvig.org/">Cewu Lu</a>
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
 
Object pose estimation is an important topic in 3D vision. Though most current state-of-the-art method that trains on real-world pose annotations achieve good results, the cost of such real-world training data is too high. In this paper, we propose a novel method for sim-to-real pose estimation, which is effective on both instance-level and category-level settings. The proposed method is based on the point-pair voting scheme from CPPF to vote for object centers, orientations, and scales. Unlike naive point pairs, to enrich the context provided by each voting unit, we introduce $N$-point tuples to fuse features from more than two points. Besides, a novel vote selection module is leveraged in order to discard those `bad' votes. Experiments show that our proposed method greatly advances the performance on both instance-level and category-level scenarios. Our method further narrows the gap between sim-to-real and real-training methods by generating synthetic training data online efficiently, while all previous sim-to-real methods need to generate data offline, because of their complex background synthesizing or photo-realistic rendering.

## Code
v0.0.1: Initial release. Support training and evaluation on NOCS dataset. 
  - We follow the same data processing pipeline and dependency setup as [CPPF](https://github.com/qq456cvb/CPPF).
  - Bug may exist, as the code is refactored with Pytorch Lightning. Performance may vary from the original implementation.
