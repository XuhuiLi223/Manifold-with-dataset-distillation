# [AAAI 2024] M3D: Dataset Condensation by Minimizing Maximum Mean Discrepancy

## [Paper](https://arxiv.org/abs/2312.15927)
Previous Distribution-Matching based methods in Dataset Condensation/Distillation naively align only the first-order moment of the feature representations of real and synthetic data. However, identical first-order moments does not guarantee identical distributions, as shown below.
![image](figs/intro.png)
To address this issue, we further embed the feature representations into a reproducing kernel Hilbert space (RKHS), where we can easily align infinite order moments, leading to a more distribution-aligned synthetic set.
![image](figs/framework.png)
![image](figs/loss_curve.png)
## Getting Started
1. Change the **data path** and **result path** in configs/dataset/IPCxxx.yaml
2. Perform the condensation process
```
python condense_m3d.py --cfg ./configs/CIFAR-10/IPC50.yaml
```
3. Condensing the ImageNet-100 dataset needs a multi-processing version
```
python condense_m3d_multiprocess.py --cfg ./configs/ImageNet-100/IPC10.yaml --phase 0 --nclass_sub 20
python condense_m3d_multiprocess.py --cfg ./configs/ImageNet-100/IPC10.yaml --phase 1 --nclass_sub 20
python condense_m3d_multiprocess.py --cfg ./configs/ImageNet-100/IPC10.yaml --phase 2 --nclass_sub 20
python condense_m3d_multiprocess.py --cfg ./configs/ImageNet-100/IPC10.yaml --phase 3 --nclass_sub 20
python condense_m3d_multiprocess.py --cfg ./configs/ImageNet-100/IPC10.yaml --phase 4 --nclass_sub 20
```
## Evaluation
We provide a script to evaluate the condensed images
```
python evaluate_synset.py --dataset cifar10 --data_dir <path to your CIFAR dataset> \
                          --syn_data_dir <path to the saved condensed images> \
                          --dsa_strategy color_crop_flip_scale_rotate \
                          --epochs 1000 \
```

## Acknowledgement
Our code is built upon [IDC](https://github.com/snu-mllab/efficient-dataset-condensation)
## Citation
If you find our code useful for your research, please cite our paper.
```
@inproceedings{zhang2024m3d,
      title={M3D: Dataset Condensation by Minimizing Maximum Mean Discrepancy}, 
      author={Hansong Zhang and Shikun Li and Pengju Wang and Dan Zeng and Shiming Ge},
      year={2024},
      booktitle={The 38th Annual AAAI Conference on Artificial Intelligence (AAAI)}
}
```