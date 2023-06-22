# An Implementation of Kernel Contrastive Learning

## Python requirements
The code is tested with python==3.10.4 , pytorch==1.12.0, torchvision==0.13.0, cuda==11.3, numpy==1.21.6.

## How to Run
1. Create and acitivate your environment on your machine based on "Python Requirements" above.
2. Prepare your datasets (note that the code supports CIFAR-10, STL-10, and ImageNet-100).
3. Train an encoder model by following the section "Training" below.
4. Perform linear evaluation with the pretrained model by following the section "Linear Classification" below.

### Training
The following command is for CIFAR-10 with GKCL. Please replace `[data location]` with the location of your dataset. Note that our code is intended for the experiments using only single GPU. If your machine has multiple GPUs, then please specify which GPU to use (e.g., by setting the enviromental variable `CUDA_VISIBLE_DEVICES` to the id of the GPU you want to use). You can also pretrain your model with QKCL by replacing `gaussian` in the option "type" with `quadratic`. Note that QKCL has only the weight (the option "weight").
```
python main_pretrain.py \
    -a resnet18 \
    -b 256 \
    --dataset cifar10 \
    --image_size 32 \
    --epochs 800 \
    --init_lr 0.0005 \
    --base_lr 0.05 \
    --warmup_epochs 10 \
    --lr_scale linear \
    --wd 0.0005 \
    --momentum 0.9 \
    --type gaussian \
    --band_width 1.0 \
    --weight 8.0 \
    --conv1_type cifar \
    --no_maxpool \
    --proj_layer 2 \
    --dim 512 \
    --optimizer sgd \
    --log_path log/gaussian/cifar10_resnet18_800epochs_bw1_w8 \
    [data location]
```


### Linear Evaluation
Please replace `[data location]` with the location of your dataset.
Remark: Our code is intended for the experiments using only single GPU. If your machine has multiple GPUs, then please specify which GPU to use (e.g., by setting the enviromental variable `CUDA_VISIBLE_DEVICES` to the id of the GPU you want to use). You can also perform linear evaluation with your pretrained model with QKCL by replacing `gaussian` in the option "type" with `quadratic`.
```
python main_lincls.py \
    -a resnet18 \
    -b 256 \
    --conv1_type cifar \
    --no_maxpool \
    --type gaussian \
    --dataset cifar10 \
    --image_size 32 \
    --epochs 100 \
    --lr 30.0 \
    --momentum 0.9 \
    --wd 0.0 \
    --pretrained log/gaussian/cifar10_resnet18_800epochs_bw1_w8/checkpoint_0799.pth.tar \
    --log_path log/gaussian/cifar10_resnet18_800epochs_bw1_w8 \
    [data location]
```
