<h1 align="center">
CPPF++: Uncertainty-Aware Sim2Real Object Pose Estimation 
 
 by Vote Aggregation

 (Under Review)
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
 
Object pose estimation constitutes a critical area within the domain of 3D vision. While contemporary state-of-the-art methods that leverage real-world pose annotations have demonstrated commendable performance, the procurement of such real-world training data incurs substantial costs. This paper focuses on a specific setting wherein only 3D CAD models are utilized as a priori knowledge, devoid of any background or clutter information. We introduce a novel method, CPPF++, designed for sim-to-real pose estimation. This method builds upon the foundational point-pair voting scheme of CPPF, reconceptualizing it through a probabilistic lens. To address the challenge of voting collision, we model voting uncertainty by estimating the probabilistic distribution of each point pair within the canonical space. This approach is further augmented by iterative noise filtering, employed to eradicate votes associated with backgrounds or clutters.
Additionally, we enhance the context provided by each voting unit by introducing $N$-point tuples. In conjunction with this methodological contribution, we present a new category-level pose estimation dataset, DiversePose 300. This dataset is specifically crafted to facilitate a more rigorous evaluation of current state-of-the-art methods, encompassing a broader and more challenging array of real-world scenarios.
Empirical results substantiate the efficacy of our proposed method, revealing a significant reduction in the disparity between simulation and real-world performance. 

## Update Logs
- Major update on methods (2023/09/06), check our updated Arxiv paper for more details. Now CPPF++ has a much better performance on both NOCS REAL275 and YCB-Video, **using only synthetic CAD models** for training. Code coming soon.

## Code
v0.0.1: Initial release. Support training and evaluation on NOCS dataset. 
  - We follow the same data processing pipeline and dependency setup as [CPPF](https://github.com/qq456cvb/CPPF).
  - This implementation is for the legacy method (arxiv v1).
