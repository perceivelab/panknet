<div align="center"> 
  
# Hierarchical 3D Feature Learning for Pancreas Segmentation
Federica Proietto Salanitri, Giovanni Bellitto, Ismail Irmakci, Simone Palazzo, Ulas Bagci, Concetto Spampinato

[![Paper](http://img.shields.io/badge/paper-arxiv.2109.01667v1-B31B1B.svg)](https://arxiv.org/abs/2109.01667)
[![Conference](http://img.shields.io/badge/MLMI-2021-4b44ce.svg)](https://link.springer.com/chapter/10.1007%2F978-3-030-87589-3_25)
</div>

# Overview

Novel 3D fully convolutional deep network for automated pancreas segmentation from both MRI and CT scans. The proposed model consists of a 3D encoder that learns to extract volume features at different scales; features taken at different points of the encoder hierarchy are then sent to multiple 3D decoders that individually predict intermediate segmentation
maps. Finally, all segmentation maps are combined to obtain a unique detailed segmentation mask. The model outperforms existing methods on CT pancreas segmentation on publicly available NIH Pancreas-CT dataset (consisting of 82 contrast-enhanced CTs), obtaining an average Dice score of about 88%. Furthermore, yields promising segmentation performance on a very challenging private MRI dataset, consisting of 40 MRI scans (average Dice score is about 77%).

# Method

<p align = "center"><img src="img/PankNet.png" width="600" style = "text-align:center"/></p>

## Examples
<p align = "center"><img src="img/SegmentationImage.PNG" width="600" style = "text-align:center"/></p>

## Notes

- As Feature Extractor, PankNet employs S3D pretained on Kinetics-400 dataset. The S3D weights can be downloaded from [here](https://github.com/kylemin/S3D).

### Citation   
```
@InProceedings{10.1007/978-3-030-87589-3_25,
author="Proietto Salanitri, Federica
and Bellitto, Giovanni
and Irmakci, Ismail
and Palazzo, Simone
and Bagci, Ulas
and Spampinato, Concetto",
editor="Lian, Chunfeng
and Cao, Xiaohuan
and Rekik, Islem
and Xu, Xuanang
and Yan, Pingkun",
title="Hierarchical 3D Feature Learning forPancreas Segmentation",
booktitle="Machine Learning in Medical Imaging",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="238--247",
isbn="978-3-030-87589-3"
}
```   
