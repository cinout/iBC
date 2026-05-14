# iBC: Identify Backdoor Channels in Contrastive Pre-trained Encoders

## Install Packages

Install the packages listed in `requirements.txt`

## Prepare Datasets

All datasets are stored in the `datasets/` folder.

### CIFAR-10/100

Both datasets are automatically downloaded during the first run.

### ImageNet-100

- Download the ImageNet-1k dataset from https://www.image-net.org/download.php
- The folder structure for ImageNet should be
  - train data: `datasets/Imagenet100/train/[class_name]/[image_name.jpg]`
  - evaluation data: `datasets/Imagenet100/val/[class_name]/[image_name.jpg]`
- In the `datasets` folder, there are `imagenet100_train_clean_filelist.txt` and `imagenet100_val_clean_filelist.txt` files, listing the train and test images used in the experiments. In these files, replace the `[Your_Absolute_Path]` with the absolute path your dataset is located at.

## Pre-train Poisoned Encoders

Here we specify some argument choices used in the experiments:

- When `trigger_type` is `ftrojan`, the `magnitude_train` is set to 50.0 for CIFAR-10/100 and to 300.0 for ImageNet-100, the `magnitude_val` is set to 100.0 for CIFAR-10/100 and to 300.0 for ImageNet-100.
- The `image_size` is set to 32 for CIFAR-10/100 and to 64 for ImageNet-100.
- The `target_class` is set to 0 for CIFAR-10/100 and to 26 for ImageNet-100.

Below are some examples of pre-training poisoned encoders for different cases.

### BYOL + FTrojan + CIFAR-100

```
python -u main_train.py \
    --method byol \
    \
    --trigger_type ftrojan \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    \
    --dataset cifar100 \
    --target_class 0 \
    --image_size 32 \
```

### SimCLR + HTBA + CIFAR-10

```
python -u main_train.py \
    --method simclr \
    \
    --trigger_type htba \
    \
    --dataset cifar10 \
    --target_class 0 \
    --image_size 32 \
```

### MoCoV2 + FTrojan + ImageNet100

```
python -u main_train.py \
    --method mocov2 \
    \
    --trigger_type ftrojan \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    \
    --dataset imagenet100 \
    --target_class 26 \
    --image_size 64 \
```

## Pre-trained Poisoned Encoder and Linear Classifier Checkpoints

After pre-training, the poisoned encoder and linear classifier are saved under the folder `Experiments/[timestamp]_[case]_[seed]/`. The encoder is `encoder.pth.tar` and linear classifier is `linear.pth.tar`.

## Run iBC Defense

To run iBC defense for each case, add the following command-line options to each case-level script above. The `find_channels_from_n_poison_samples` denotes the number of estimated poisoned images, and can be of any small positive integer.

```
--use_ibc \
--find_channels_from_n_poison_samples 2 \
--pretrained_ssl_model [path_to_encoder_checkpoint]
--pretrained_linear_model [path_to_linear_classifier_checkpoint]
```
