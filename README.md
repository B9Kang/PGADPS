# Prior-aware and Context-guided Group Sampling for Active Probabilistic Subsampling (ICLR 2026)

---

## 🔍 Overview

This repository provides the **official PyTorch implementation** of  
**ICLR 2026 paper "Prior-aware and Context-guided Group Sampling for Active Probabilistic Subsampling"**

Our PGA-DPS (Prior-aware and Group-based Active Deep Probabilistic Subsampling) extends Active Deep Probabilistic Subsampling (A-DPS) by:
- incorporating **deterministic prior-aware sampling** learned from training data, and
- enabling **group-based (top-k) active sampling**, which leads to smoother optimization and improved robustness.

The method is evaluated on **three tasks**:
- **Classification** (MNIST, CIFAR-10)
- **MRI reconstruction** (fastMRI knee)
- **Hyperspectral image (HSI) segmentation** (AeroRIT)

Across all tasks, PGA-DPS consistently outperforms DPS, A-DPS, and other state-of-the-art sampling strategies.

---

## 🧠 Method Highlights

- **Prior-aware deterministic sampling**: exploits dataset-level priors
- **Context-guided active sampling**: adapts sampling to each input instance
- **Group-based top-k sampling**: improves optimization stability via smoother loss landscapes
- **End-to-end training**: jointly optimizes sampling strategy and downstream task model

---

## 📌 Pre-Trained models

```text
.
├── MRI_Reconstruction/
│   ├── checkpoints/
│   │   ├── ADPS_26lines_0seed.tar
│   │   ├── DPS_26lines_0seed.tar
│   │   ├── PGADPS_26lines_0seed.tar
│   │   ├── ...

```
Test code

analyseCheckpoint.py -sampling PGADPS -save_name PGADPS_26lines_30Ps_30As_0seed -Ps 30 -As 30

analyseCheckpoint.py -sampling ADPS -save_name ADPS_26lines_0seed

analyseCheckpoint.py -sampling DPS -save_name DPS_26lines_1seed

## Implementation 
**1. Classification on MNIST dataset [1]**

MNIST_Classification/main.py -sampling 'PGADPS' -percentage 2 -Ps 60 -As 20 -seed 0


**2. Classification on CIFAR10 dataset [2]**

CIFAR_Classification/main.py -sampling 'PGADPS' -percentage 2 -Ps 10 -As 20 -seed 0


**3. MRI reconstruction on FastMRI dataset [3]**

To perform MRI reconstruction, you need to download the single-coil knee training and validation data from the fastMRI website (https://fastmri.med.nyu.edu/). After downloading the dataset, place it in "my_path" as shown below and run the 'preprocessing.py':

```text
my_path/
├── knee_singlecoil_train/
│   ├── file1000001.h5  
│   ├── file1000002.h5  
│   ├── ...
│
├── knee_singlecoil_val/
│   ├── file1000000.h5  
│   ├── file1000007.h5  
│   ├── ...
```

MRI_Reconstruction/preprocessing.py -path my_path


Then, run the MRI reconstruction code:

MRI_Reconstruction/main.py -sampling 'PGADPS'


To test the MR reconstruction with pre-trained models (saved in checkpoints), run

MRI_Reconstruction/analyseCheckpoint.py -sampling 'PGADPS' -Ps 30 -As 30 -seed 0



**4. Hyperspectral image (HSI) segmentation on AeroRIT dataset [4]**

To perform HSI segmentation, you need to download the HSI data from the AeroRIT website (https://github.com/aneesh3108/AeroRIT). After downloading the dataset, place it in "Aerial Data/Collection" as shown below and run the 'sampling_data.py':

```text
Aerial Data/
├── Collection/
│   ├── image_hsi_radiance.tif  
│   ├── image_hsi_reflectance.tif 
│   ├── image_labels.tif
│   ├── image_rgb.tif
```

Hyperspectral/sampling_data.py


Then, run the HSI segmentation code:

Hyperspectral/train_DPS.py -sampling 'PGADPS' -Ps 80 -As 20 -seed 0


To test the HSI segmentation with pre-trained models (saved in savedmodels), run

Hyperspectral/test_DPS.py -sampling 'PGADPS' -network_saved_name PGADPS_5bands_80Ps_20As_0seed



_______________________________________________
References

[1] Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278–2323.

[2] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[3] Jure Zbontar, et al. "fastMRI: An open dataset and benchmarks for accelerated MRI." arXiv preprint arXiv:1811.08839 (2018).

[4] Rangnekar, A., Mokashi, N., Ientilucci, E. J., Kanan, C., and Hoffman, M. J. (2020). "Aerorit: A new scene for hyperspectral image analysis." IEEE Transactions on Geoscience and Remote Sensing, 58(11):8116–8124.


--- 

## 🙏 Acknowledgment

Our codebase is heavily based on the excellent Deep probabilistic Subsampling repository by [IamHuijben](https://github.com/IamHuijben/Deep-Probabilistic-Subsampling). We sincerely appreciate their contribution to the community.

## 📚 Citation

If you find our work useful or valuable for your research, please consider citing us:

```bibtex
@inproceedings{
  kang2026prioraware,
  title={Prior-aware and Context-guided Group Sampling for Active Probabilistic Subsampling},
  author={Kang, Beomgu and Seo, Hyunseok},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
