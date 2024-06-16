# RobustSAM: Segment Anything Robustly on Degraded Images (CVPR 2024 Highlight)

Official repository for RobustSAM: Segment Anything Robustly on Degraded Images



[Project Page](https://robustsam.github.io/) | [Paper]() | [Video]() | [Dataset]()



## Updates
- Feb 2024: ✨ RobustSAM was accepted into CVPR 2024!


## Setup
1) Use "nvidia-smi" command to check your CUDA version.
2) Replace the CUDA version with yours in command below.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[$YOUR_CUDA_VERSION]
# For example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 # cu117 = CUDA_version_11.7
```


## Reference
If you find this work useful, please consider citing us!
```python
@inproceedings{chen2024robustsam,
  title={RobustSAM: Segment Anything Robustly on Degraded Images},
  author={Chen, Wei-Ting and Vong, Yu-Jiet and Kuo, Sy-Yen and Ma, Sizhou and Wang, Jian},
  journal={CVPR},
  year={2024}
}
```


## Acknowledgements
We thank the authors of [SAM](https://github.com/facebookresearch/segment-anything) from which our repo is based off of.

