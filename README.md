<div align="center">
<h2>BEVDilation: LiDAR-Centric Multi-Modal Fusion for 3D Object Detection</h2>

<h3 align="center"><b>AAAI 2026</b></h3>

[**Guowen Zhang**](https://scholar.google.com/citations?user=DxcLKZIAAAAJ&hl=en) · [**Chenhang He**](https://scholar.google.com/citations?user=dU6hpFUAAAAJ&hl=en) · [**Liyi Chen**](https://scholar.google.com/citations?user=nMev-10AAAAJ&hl=zh-CN) · [**Zhang Lei**](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>&dagger;</sup>
<br>

The Hong Kong Polytechnic University 
<br>
&dagger;Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2512.02972)

</div>

Our BEVDilation achieves state-of-the-art performance on nuScene datasets. It prioritizes LiDAR information in the multi-modal fusion, achieving effective and robust fusion.

## 🔥News
-[25-11-24] BEVDilation released on [arxiv](https://arxiv.org/pdf/2512.02972)   
-[25-11-24] BEVDilation is accepted by **AAAI26**!

## 📘TODO
- [x] Release the [arXiv](https://arxiv.org/pdf/2512.02972) version.
- [x] Clean up and release the code.
<!-- - [x] Release code of Waymo.
- [ ] Release code of NuScenes.
- [ ] Release code of ERFs visualization.
- [ ] Merge Voxel Mamba to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). -->

## 🏆Main Results

#### nuScene Dataset
Validation set  
|  Model  | mAP | NDS | mATE | mASE | mAOE | mAVE| mAAE | ckpt |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|
|  [BEVDilation]() | 73.0 | 75.0 | 26.9 | 24.7 | 28.6 | 17.7 | 17.3 | [ckpt]()| 

Test set  
|  Model  | mAP | NDS | mATE | mASE | mAOE | mAVE| mAAE | Leaderboard | Submission |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|--------|
|  [BEVDilation]() | 73.1 | 75.4 | 24.8 | 23.4 | 33.8 | 17.8 | 11.7| [leaderboard]()| [Submission]()|  


BEVDilation's result on nuScenes compared with other leading methods.
All the experiments are evaluated on an NVIDIA A6000 GPU with the same environment.
We hope that our BEVDilation can provide a potential LiDAR-centric solution for efficiently handling multi-modal fusion for 3D tasks.
<div align="left">
  <img src="docs/PVsS.png" width="500"/>
</div>

## 🚀Usage
### Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for installation.

### Dataset Preparation

```
BEVDilation/
├── data/
│   └── nuscenes/
│       ├── samples/          # Sensor data (keyframes)
│       ├── sweeps/           # Sensor data (intermediate frames)
│       ├── maps/             # Map data (optional)
│       ├── v1.0-trainval/    # Metadata for train and val splits
│       ├── v1.0-test/        # Metadata for test split
|       ├── bevdetv3-nuscenes_gt_database/
|       ├── bevdetv3-nuscenes_dbinfos_train.pkl       
|       ├── bevdetv3-nuscenes_infos_train.pkl
|       └── bevdetv3-nuscenes_infos_val.pkl
```

### Generate Hilbert Template, following [Voxel Mamba](https://github.com/gwenzhang/Voxel-Mamba)
```
cd data
mkdir hilbert
python ./tools/create_hilbert_curve_template.py
```
You can also download Hilbert Template files from [Google Drive](https://drive.google.com/drive/folders/1MF73ZP50fw4jNlDoh_bLiE0Z-GQVLjMu?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1gCaNK9qaLRSZNBB3FPujuQ) (code: mwd4).

### Training
```
# multi-gpu training
cd tools
./tools/dist_train.sh configs/bevdilation/bevdilation.py 8
```

### Test
```
# multi-gpu testing
./tools/dist_test.sh ./bevdilation/bevdilation.py ./checkpoint_path 8 --eval mAP
```

## Citation
Please consider citing our work as follows if it is helpful.
```
@article{zhang2024bevdilation,
  title={BEVDilation: LiDAR-Centric Multi-Modal Fusion for 3D Object Detection},
  author={Zhang, Guowen and He, Chenhang and Chen Liyi and Zhang, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Acknowledgments
BEVDilation is based on [DAL](https://github.com/HuangJunJie2017/BEVDet).  
We also thank the Voxel Mamba, DAL, OpenPCDet, and MMDetection3D authors for their efforts.



