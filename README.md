## Video Contrastive Learning with Global Context (VCLR)

This is the official PyTorch implementation of our [VCLR paper](https://arxiv.org/abs/2108.02722).

```
@article{kuang2021vclr,
  title={Video Contrastive Learning with Global Context},
  author={Haofei Kuang, Yi Zhu, Zhi Zhang, Xinyu Li, Joseph Tighe, Sören Schwertfeger, Cyrill Stachniss, Mu Li},
  journal={arXiv preprint arXiv:2108.02722},
  year={2021}
}
```


## Install dependencies
- environments
  ```shell
  conda create --name vclr python=3.7
  conda activate vclr
  conda install numpy scipy scikit-learn matplotlib scikit-image
  pip install torch==1.7.1 torchvision==0.8.2
  pip install opencv-python tqdm termcolor gcc7 ffmpeg tensorflow==1.15.2
  pip install mmcv-full==1.2.7
  ```


## Prepare datasets
Please refer to [PREPARE_DATA](PREPARE_DATA.md) to prepare the datasets.


## Prepare pretrained MoCo weights
In this work, we follow [SeCo](https://arxiv.org/abs/2008.00975) and use the pretrained weights of [MoCov2](https://github.com/facebookresearch/moco) as initialization.

```shell
cd ~
git clone https://github.com/amazon-research/video-contrastive-learning.git
cd video-contrastive-learning
mkdir pretrain && cd pretrain
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar
cd ..
```


## Self-supervised pretraining

```shell
bash shell/main_train.sh
```
Checkpoints will be saved to `./results`


## Downstream tasks

### Linear evaluation
In order to evaluate the effectiveness of self-supervised learning, we conduct a linear evaluation (probing) on Kinetics400 dataset. Basically, we first extract features from the pretrained weight and then train a SVM classifier to see how the learned features perform.

```shell
bash shell/eval_svm.sh
```

- Results

  | Arch | Pretrained dataset | Epoch | Pretrained model | Acc. on K400 |
  | :------: | :-----: | :-----: | :-----: | :-----: |
  | ResNet50 | Kinetics400 | 400 | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/pretrained/vclr_k400.pth) | 64.1 |


### Video retrieval

```shell
bash shell/eval_retrieval.sh
```

- Results

  | Arch | Pretrained dataset | Epoch | Pretrained model | R@1 on UCF101 | R@1 on HMDB51 |
  | :------: | :-----: | :-----: | :-----: | :-----: | :-----: |
  | ResNet50 | Kinetics400 | 400 | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/pretrained/vclr_k400.pth) | 70.6 | 35.2 |
  | ResNet50 | UCF101 | 400 | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/pretrained/vclr_ucf.pth) | 46.8 | 17.6 |


### Action recognition & action localization

Here, we use mmaction2 for both tasks. If you are not familiar with mmaction2, you can read the [official documentation](https://mmaction2.readthedocs.io/en/latest/index.html).

#### Installation
- Step1: Install mmaction2

  To make sure the results can be reproduced, please use our forked version of mmaction2 (version: 0.11.0):
  ```shell
  conda activate vclr
  cd ~
  git clone https://github.com/KuangHaofei/mmaction2

  cd mmaction2
  pip install -v -e .
  ```
- Step2: Prepare the pretrained weights

  Our pretrained backbone have different format with the backbone of mmaction2, it should be transferred to mmaction2 format. We provide the transferred version of our K400 pretrained weights, [TSN](https://haofeik-data.s3.amazonaws.com/VCLR/pretrained/vclr_mm.pth) and [TSM](https://haofeik-data.s3.amazonaws.com/VCLR/pretrained/vclr_mm_tsm.pth). We also provide the script for transferring weights, you can find it [here](./tools/weights/README.md).

  Moving the pretrained weights to `checkpoints` directory:
  ```shell
  cd ~/mmaction2
  mkdir checkpoints
  wget https://haofeik-data.s3.amazonaws.com/VCLR/pretrained/vclr_mm.pth
  wget https://haofeik-data.s3.amazonaws.com/VCLR/pretrained/vclr_mm_tsm.pth
  ```

#### Action recognition
Make sure you have prepared the dataset and environments following the previous step. Now suppose you are in the root directory of `mmaction2`, follow the subsequent steps to fine tune the TSN or TSM models for action recognition.

For each dataset, the train and test setting can be found in the configuration files.

- UCF101
  - config file: [tsn_ucf101.py](https://github.com/KuangHaofei/mmaction2/blob/master/configs/recognition/tsn/vclr/tsn_ucf101.py)
  - train command:
    ```shell
    ./tools/dist_train.sh configs/recognition/tsn/vclr/tsn_ucf101.py 8 \
      --validate --seed 0 --deterministic
    ```
  - test command:
    ```shell
    python tools/test.py configs/recognition/tsn/vclr/tsn_ucf101.py \
      work_dirs/vclr/ucf101/latest.pth \
      --eval top_k_accuracy mean_class_accuracy --out result.json
    ```

- HMDB51
  - config file: [tsn_hmdb51.py](https://github.com/KuangHaofei/mmaction2/blob/master/configs/recognition/tsn/vclr/tsn_hmdb51.py)
  - train command:
    ```shell
    ./tools/dist_train.sh configs/recognition/tsn/vclr/tsn_hmdb51.py 8 \
      --validate --seed 0 --deterministic
    ```
  - test command:
    ```shell
    python tools/test.py configs/recognition/tsn/vclr/tsn_hmdb51.py \
      work_dirs/vclr/hmdb51/latest.pth \
      --eval top_k_accuracy mean_class_accuracy --out result.json
    ```

- SomethingSomethingV2: TSN
  - config file: [tsn_sthv2.py](https://github.com/KuangHaofei/mmaction2/blob/master/configs/recognition/tsn/vclr/tsn_sthv2.py)
  - train command:
    ```shell
    ./tools/dist_train.sh configs/recognition/tsn/vclr/tsn_sthv2.py 8 \
      --validate --seed 0 --deterministic
    ```
  - test command:
    ```shell
    python tools/test.py configs/recognition/tsn/vclr/tsn_sthv2.py \
      work_dirs/vclr/tsn_sthv2/latest.pth \
      --eval top_k_accuracy mean_class_accuracy --out result.json
    ```
- SomethingSomethingV2: TSM
  - config file: [tsm_sthv2.py](https://github.com/KuangHaofei/mmaction2/blob/master/configs/recognition/tsm/vclr/tsm_sthv2.py)
  - train command:
    ```shell
    ./tools/dist_train.sh configs/recognition/tsm/vclr/tsm_sthv2.py 8 \
      --validate --seed 0 --deterministic
    ```
  - test command:
    ```shell
    python tools/test.py configs/recognition/tsm/vclr/tsm_sthv2.py \
      work_dirs/vclr/tsm_sthv2/latest.pth \
      --eval top_k_accuracy mean_class_accuracy --out result.json
    ```

- ActivityNet
  - config file: [tsn_activitynet.py](https://github.com/KuangHaofei/mmaction2/blob/master/configs/recognition/tsn/vclr/tsn_activitynet.py)
  - train command:
    ```shell
    ./tools/dist_train.sh configs/recognition/tsn/vclr/tsn_activitynet.py 8 \
      --validate --seed 0 --deterministic
    ```
  - test command:
    ```shell
    python tools/test.py configs/recognition/tsn/vclr/tsn_activitynet.py \
      work_dirs/vclr/tsn_activitynet/latest.pth \
      --eval top_k_accuracy mean_class_accuracy --out result.json
    ```

- Results

  | Arch | Dataset | Finetuned model | Acc. |
  | :------: | :-----: | :-----: | :-----: |
  | TSN | UCF101 | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/action_recognition/mm_ucf_tsn.pth) | 85.6 |
  | TSN | HMDB51 | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/action_recognition/mm_hmdb_tsn.pth) | 54.1 |
  | TSN | SomethingSomethingV2 | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/action_recognition/mm_sthv2_tsn.pth) | 33.3 |
  | TSM | SomethingSomethingV2 | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/action_recognition/mm_sthv2_tsm.pth) | 52.0 |
  | TSN | ActivityNet | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/action_recognition/mm_anet_tsn.pth) | 71.9 |


#### Action localization
- Step 1: Follow the previous section, suppose the finetuned model is saved at `work_dirs/vclr/tsn_activitynet/latest.pth`

- Step 2: Extract ActivityNet features
  ```shell
  cd ~/mmaction2/tools/data/activitynet/

  python tsn_feature_extraction.py --data-prefix /home/ubuntu/data/ActivityNet/rawframes \
    --data-list /home/ubuntu/data/ActivityNet/anet_train_video.txt \
    --output-prefix /home/ubuntu/data/ActivityNet/rgb_feat \
    --modality RGB --ckpt /home/ubuntu/mmaction2/work_dirs/vclr/tsn_activitynet/latest.pth

  python tsn_feature_extraction.py --data-prefix /home/ubuntu/data/ActivityNet/rawframes \
    --data-list /home/ubuntu/data/ActivityNet/anet_val_video.txt \
    --output-prefix /home/ubuntu/data/ActivityNet/rgb_feat \
    --modality RGB --ckpt /home/ubuntu/mmaction2/work_dirs/vclr/tsn_activitynet/latest.pth

  python activitynet_feature_postprocessing.py \
    --rgb /home/ubuntu/data/ActivityNet/rgb_feat \
    --dest /home/ubuntu/data/ActivityNet/mmaction_feat
  ```
  Note, the root directory of ActivityNey is `/home/ubuntu/data/ActivityNet/` in our case. Please replace it according to your real directory.

- Step 3: Train and test the BMN model
  - train
    ```shell
    cd ~/mmaction2
    ./tools/dist_train.sh configs/localization/bmn/bmn_acitivitynet_feature_vclr.py 2 \
      --work-dir work_dirs/vclr/bmn_activitynet --validate --seed 0 --deterministic --bmn
    ```
  - test
    ```shell
    python tools/test.py configs/localization/bmn/bmn_acitivitynet_feature_vclr.py \
      work_dirs/vclr/bmn_activitynet/latest.pth \
      --bmn --eval AR@AN --out result.json
    ```

- Results

  | Arch | Dataset | Finetuned model | AUC | AR@100 |
  | :------: | :-----: | :-----: | :-----: | :-----: |
  | BMN | ActivityNet | [Download link](https://haofeik-data.s3.amazonaws.com/VCLR/action_localization/mm_anet_bmn.pth) | 65.5 | 73.8 |


## Feature visualization

We provide our feature visualization code at [here](./tools/feature_visualization/README.md).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.


## License

This project is licensed under the Apache-2.0 License.
