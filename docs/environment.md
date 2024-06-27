# Environment Setup

Our model is built upon [PF-Track](https://github.com/TRI-ML/PF-Track) and the environment setup can refer to [here](https://github.com/TRI-ML/PF-Track/blob/main/documents/environment.md).
The following are some key points.

1. Pytorch-related. 
`pytorch==1.9.0`, `cuda==11.1`, `cudnn==8`.

2. MMLab-related. (build from source recommended). 
`MMCV==1.4.0`, `MMDetection==v2.24.1`, `MMSegmentation==v0.20.2`, `MMDetection3d==v0.17.1`.

3. nuScenes-related. `nuscenes-devkit==1.1.7`, `motmetrics==1.1.3` (don't use higher versions of `motmetrics`, or it will cause `nuscenes-devkit` into bugs, which is a known nuScenes dependency issue.)

We thank [PF-Track](https://github.com/TRI-ML/PF-Track) for open-sourcing. 