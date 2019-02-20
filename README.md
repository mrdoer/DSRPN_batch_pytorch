# DenseSiamese-RPN

This is a PyTorch implementation of DenseSiameseRPN. This project is mainly based on [DensSiamFC](http://submit.votchallenge.net/data/vot2018/DensSiam-code-2018-06-18T02:59:51.877009.zip),[SiamRPN pytorch version](https://github.com/HelloRicky123/Siamese-RPN),[SiamFC-PyTorch](https://github.com/StrangerZhang/SiamFC-PyTorch) and [DaSiamRPN](https://github.com/foolwood/DaSiamRPN).

This repository includes training and tracking codes. 

## Data preparation:

python bin/create_dataset.py --data-dir /dataset_ssd/ILSVRC2015 --output-dir /dataset_ssd/vid15rpn_finetune

python bin/create_lmdb.py --data-dir /dataset_ssd/vid15rpn_finetune --output-dir /dataset_ssd/vid15rpn_finetune.lmdb

## Traing phase:

CUDA_VISIBLE_DEVICES=2 python bin/train_siamfc.py --data_dir /dataset_ssd/vid15rpn_large

## Reference

[1] Li B , Yan J , Wu W , et al. High Performance Visual Tracking with Siamese Region Proposal Network[C]// 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2018.

[2] Abdelpakey, Mohamed H., Mohamed S. Shehata, and Mostafa M. Mohamed. "DensSiam: End-to-End Densely-Siamese Network with Self-Attention Model for Object Tracking." In International Symposium on Visual Computing, pp. 463-473. Springer, Cham, 2018.(https://arxiv.org/abs/1809.02714)
