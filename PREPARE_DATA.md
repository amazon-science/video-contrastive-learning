## [Kinetics-400](https://deepmind.com/research/open-source/kinetics)

- The final data structure should look like
```
kinetics400_30fps_frames/
├── train/
│   ├── abseiling/
│   │   ├──0347ZoDXyP0_000095_000105
│   │   │  ├──frame_00001.jpg
│   │   │  ├──...
│   │   ├──...
│   ├──...
├── val/
│   ├── abseiling/
│   │   ├──0wR5jVB-WPk
│   │   │  ├──frame_00001.jpg
│   │   │  ├──...
│   │   ├──...
│   ├──...
├── train.txt
├── val.txt
```

### Prepare Kinetics-400 dataset step-by-step

- step 1: Download raw videos either from [Academic Torrents](https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26) or [CVDF](https://github.com/cvdfoundation/kinetics-dataset). Suppose the videos are stored at `~/data/kinetics400`.

- step 2: Extract frames from raw videos

  We fix each video to 300 frames through three rules:
  - if number of frames = 300 frames: keep it
  - if number of frames > 300 frames: drop the image after the 300th frame
  - if number of frames < 300 frames: duplicate the last frame until it fills up to 300 frames

  To extract frames of each video, we use `ffmpeg`:
  ```shell
  sudo apt-get install ffmpeg
  ```

  Follow the command below to extract frames, for example:
  ```shell
  python ./tools/data/k400/extract_frames.py --source_dir ~/data/kinetics400/train_256 --target_dir ~/data/kinetics400_30fps_frames/train
  python ./tools/data/k400/extract_frames.py --source_dir ~/data/kinetics400/val_256 --target_dir ~/data/kinetics400_30fps_frames/val
  ```

  - (optional) In case you want to extract frames faster in parallel, please follow
    ```shell
    wget https://www.parallelpython.com/downloads/pp/pp-1.6.4.4.zip
    unzip pp-1.6.4.4.zip && cd pp-1.6.4.4
    python setup.py install && cd ..

    python ./tools/data/k400/extract_frames_parallel.py --source_dir ~/data/kinetics400/train_256 --target_dir ~/data/kinetics400_30fps_frames/train
    python ./tools/data/k400/extract_frames_parallel.py --source_dir ~/data/kinetics400/val_256 --target_dir ~/data/kinetics400_30fps_frames/val
    ```

- step 3: Download train/val split files,
  ```shell
  cd ~/data/kinetics400_30fps_frames
  wget https://yzaws-data-log.s3.amazonaws.com/data/Kinetics/k400_train.txt
  wget https://yzaws-data-log.s3.amazonaws.com/data/Kinetics/k400_val.txt
  mv k400_train.txt train.txt && mv k400_val.txt val.txt
  ```


## [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

Please refer [gluoncv tutorials](https://cv.gluon.ai/build/examples_datasets/ucf101.html) to prepare this dataset.

- The final data structure should look like
```
ucf101/
├── rawframes/
│   ├── ApplyEyeMakeup/
│   │   ├──v_ApplyEyeMakeup_g01_c01/
│   │   │  ├──img_00001.jpg
│   │   │  ├──...
│   │   ├──...
│   ├──...
├── annotations/
│   ├── classInd.txt
│   ├── testlist01.txt
│   ├── testlist02.txt
│   ├── testlist03.txt
│   ├── trainlist01.txt
│   ├── trainlist02.txt
│   ├── trainlist03.txt
├── ucf101_train_split_1_rawframes.txt
├── ucf101_train_split_2_rawframes.txt
├── ucf101_train_split_3_rawframes.txt
├── ucf101_val_split_1_rawframes.txt
├── ucf101_val_split_2_rawframes.txt
├── ucf101_val_split_3_rawframes.txt
```


## [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

Please refer [gluoncv tutorials](https://cv.gluon.ai/build/examples_datasets/hmdb51.html) to prepare this dataset.

- The final data structure should look like
```
hmdb51/
├── rawframes/
│   ├── brush_hair/
│   │   ├──April_09_brush_hair_u_nm_np1_ba_goo_0/
│   │   │  ├──img_00001.jpg
│   │   │  ├──...
│   │   ├──...
│   ├──...
├── annotations/
│   ├── classInd.txt
│   ├── brush_hair_test_split1.txt
│   ├── ...
├── hmdb51_train_split_1_rawframes.txt
├── hmdb51_train_split_2_rawframes.txt
├── hmdb51_train_split_3_rawframes.txt
├── hmdb51_val_split_1_rawframes.txt
├── hmdb51_val_split_2_rawframes.txt
├── hmdb51_val_split_3_rawframes.txt
```


## [SomethingSomethingV2](https://20bn.com/datasets/something-something)

Please refer [gluoncv tutorials](https://cv.gluon.ai/build/examples_datasets/somethingsomethingv2.html) to prepare this dataset.

- The final data structure should look like
```
sthv2/
├── rawframes/
│   ├──...
│   ├── 14876/
│   │   ├──img_00001.jpg
│   │   ├──...
│   ├──...
├── annotations/
│   ├── something-something-v2-labels.json
│   ├── something-something-v2-test.json
│   ├── something-something-v2-train.json
│   ├── something-something-v2-validation.json
├── sthv2_train_list_rawframes.txt
├── sthv2_val_list_rawframes.txt
```


## [ActivityNet](http://activity-net.org/)

- The final data structure should look like
```
ActivityNet/
├── videos/
│   ├──...
│   ├── 00018--lj-VovhJcPA.mp4
│   ├── 00018--lk2niPrG3y8.webm
│   ├──...
├── rawframes/
│   ├──...
│   ├── 00015--lnHdEtuXU8w/
│   │   ├──img_00000.jpg
│   │   ├──...
│   ├──...
├── anet_anno_train.json
├── anet_anno_val.json
├── anet_train_clip.txt
├── anet_train_video.txt
├── anet_val_clip.txt
├── anet_val_video.txt
```

### Prepare ActivityNet dataset step-by-step

- step 1: Download raw videos using [official crawler](https://github.com/activitynet/ActivityNet). Suppose the videos are stored at `~/data/ActivityNet`.

- step 2: Extract frames

  - install denseflow: please refer to mmaction2 tutorials [install.md](https://mmaction2.readthedocs.io/en/latest/install.html) to install [denseflow](https://github.com/open-mmlab/denseflow)

    - add environment variables
      ```shell
      echo 'export ZZROOT=$HOME/app' >> ~/.bashrc
      echo 'export PATH=$ZZROOT/bin:$PATH' >> ~/.bashrc
      echo 'export LD_LIBRARY_PATH=$ZZROOT/lib:$ZZROOT/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
      ```
    - install denseflow by [https://github.com/innerlee/setup](https://github.com/innerlee/setup), please make sure each shell scripts could be executed correctly (maybe you lack some dependencies).
      ```shell
      cd ~
      git clone https://github.com/innerlee/setup.git
      cd setup

      sudo apt-get install autoconf cmake yasm
      ./zznasm.sh
      ./zzyasm.sh
      ./zzlibx264.sh
      ./zzlibx265.sh
      ./zzlibvpx.sh
      ./zzffmpeg.sh
      ./zzopencv.sh
      export OpenCV_DIR=$ZZROOT
      ./zzboost.sh
      ./zzdenseflow.sh
      ```
  - extract frames
    - make sure you install mmaction2, refer to [README.md](./README.md)
    - use bash script below to extract frames
      ```shell
      cd mmaction2/tools/data/activitynet/
      bash extract_rgb_frames.sh
      ```
      Note, you need to change the **raw videos path** and the **target frames path** in `extract_rgb_frames.sh` file accordingly.

- step 3: Prepare annotation files
  ```shell
  cd ~/data/ActivityNet
  wget https://yzaws-data-log.s3.amazonaws.com/data/activitynet/anet_anno.zip
  unzip anet_anno.zip
  ```
