# RobustSAM: Segment Anything Robustly on Degraded Images (CVPR 2024 Highlight)

Official repository for RobustSAM: Segment Anything Robustly on Degraded Images



[Project Page](https://robustsam.github.io/) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.html) | [Video](https://www.youtube.com/watch?v=Awukqkbs6zM) | [Dataset](https://robustsam.github.io/)


## Updates
- Feb 2024: ✨ RobustSAM was accepted into CVPR 2024!
- June 2024: ✨ Inference code of RobustSAM was released!

## Introduction

## Setup
1) Create a conda environment and activate it.
```
conda create --name robustsam python=3.10 -y
conda activate robustsam
```
2) Clone and enter into repo directory.
```
git clone https://github.com/robustsam/RobustSAM
cd RobustSAM
```
3) Use command below to check your CUDA version.
```
nvidia-smi
```
4) Replace the CUDA version with yours in command below.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[$YOUR_CUDA_VERSION]
# For example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 # cu117 = CUDA_version_11.7
```
5) Install remaining dependencies
```
pip install -r requirements.txt
```

6) Download pretrained RobustSAM checkpoint.
```
wget https://drive.google.com/file/d/197EEnWYvchupfJrK44-ki3wDhv3-UuGv/view?usp=sharing
```

## Demo
We have prepared some images im **demo_images** folder for demo purpose. Besides, two prompting modes are available (box prompts and point prompts).
- For box prompt:
```
python eval.py --bbox
```
- For point prompt:
```
python eval.py
```
In default, demo results will be saved to **demo_result/[$PROMPT_TYPE]**.

## Visual Comparison
<table>
  <tr>
    <td><img src="visual_demo/1.gif" width="250"></td>
    <td><img src="visual_demo/2.gif" width="250"></td>
  </tr>
  <tr>
    <td><img src="visual_demo/4.gif" width="250"></td>
    <td><img src="visual_demo/5.gif" width="250"></td>
  </tr>
</table>

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

