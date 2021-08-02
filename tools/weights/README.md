# Transfer VCLR weights format to different frameworks

## VCLR to mmaction2
Here, we transfer the vclr weights format to mmaction2 format through three steps:

### Step1: download TSN and TSM template weights of mmaction2
Follow the below command to download the template weightf files:  
```shell
cd ~/VCLR/weights
aws s3 cp s3://haofeik-data/VCLR/mm_templates/ ./ --recursive
```

### Step2: replace backbone values of the template weights with VCLR weight
For example, vclr weights path is `vclr_k400.pth`, and mmaction2 template weights path is `mm_tsn_template.pth` and `mm_tsm_template.pth`

- transfer weights for mmaction2 TSN model:
  ```shell
  python vclr2mm.py --mm-weights mm_tsn_template.pth --vclr-weights vclr_k400.pth
  ```
  
- transfer weights for mmaction2 TSM model:
  ```shell
  python vclr2mm.py --mm-weights mm_tsm_template.pth --vclr-weights vclr_k400.pth --tsm
  ```
  
The transfered weights  `vclr_mm.pth` or `vclr_mm_tsm.pth` will be stored at same directory.

## VCLR to torchvision
Most applications will use the ResNet-50 through torchvision, you can follow the below tutorials to transfer VCLR pretrained weights to torchvision format.

For example, vclr weights path is `vclr_k400.pth`,
```shell
python vclr2torchvision.py --vclr-weights vclr_k400.pth
```

The transfered weights  `vclr_torch.pth` will be stored at same directory.
