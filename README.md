# TDFA: A translation and diffusion-based feature augmentation framework for long-tailed visual recognition



## About this repository
The repo is implemented based on [https://github.com/w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm). Currently it supports the training for four datasets namely CIFAR10(LT) and CIFAR100(LT) under following three mechanisms:

1. Regular (conditional or unconditional) diffusion model training
2. Class-balancing model training
3. Class-balancing model finetuning based on a regular diffusion model

## Running the Experiments
We provide mainly the scripts for trianing and evaluating the CIFAR100LT dataset.
To run the code, please change the argument 'root' to the path where the dataset is downloaded.

## Files used in evaluation

Please find the [features for cifar 100 and cifar 10](https://drive.google.com/drive/folders/1Y89vu9DGiQsHl8YvwMrr_7UT4p4Pg_wV?usp=sharing) used in precision/recall/f_beta metrics. Put them in the stats folder and the codes will be ready to run. Note that those two metrics will only be evaluated if the number of samples is 50k otherwise it returns 0.

### Datasets:
cifar10/100-LT: [CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html)

imagenet-lt：[KITSCH/miniimagenet-LT at main (huggingface.co)](https://huggingface.co/datasets/KITSCH/miniimagenet-LT/tree/main)

inaturalist2018:  [inat_comp/2018 at master · visipedia/inat_comp (github.com)](https://github.com/visipedia/inat_comp/tree/master/2018)



### Train a model
* Regular conditional diffusion model training, supporting the classifier-free guidance (cfg) sampling
    ```
    python main.py --train  \
            --flagfile ./config/cifar100.txt --parallel \
            --logdir ./logs/cifar100lt_ddpm --total_steps 300001 \
            --conditional \
            --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
            --batch_size 64 --save_step 100000 --sample_step 50000 \
            --cfg
    ```

* Class-balancing model training without ADA augmentation
    ```
    python main.py --train  \
            --flagfile ./config/cifar100.txt --parallel \
            --logdir ./logs/cifar100lt_ltdm --total_steps 300001 \
            --conditional \
            --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
            --batch_size 48 --save_step 100000 --sample_step 50000 \
            --cb --tau 1.0
    ```

* Class-balancing model training with ADA augmentation
    ```
    python main.py --train  \
            --flagfile ./config/cifar100.txt --parallel \
            --logdir ./logs/cifar100lt_ltdm_augm --total_steps 500001 \
            --conditional \
            --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
            --batch_size 48 --save_step 100000 --sample_step 50000 \
            --cb --tau 1.0 --augm
    ```

* Class-balancing model finetuning: finetune a DDPM model(ckpt of 200000 steps which the classifier-free guidance (cfg) sampling) based on LTDM approach
    ```
    python main.py --train  \
            --flagfile ./config/cifar100.txt --parallel \
            --logdir ./logs/cifar100lt_ddpm --total_steps 100001 \
            --conditional \
            --data_type cifar100lt --imb_factor 0.01 --img_size 32 \
            --batch_size 48 --save_step 50000 --sample_step 50000 \
            --cb --tau 1.0 \
            --finetune --finetuned_logdir cifar100lt_ltdm_finetune --ckpt_step 200000
    ```

### Evaluate a model
* Sample images and evaluate for the 4 models above.

    ```
    python main.py \
        --flagfile ./logs/cifar100lt_ddpm/flagfile.txt \
        --logdir ./logs/cifar100lt_ddpm \
        --fid_cache ./stats/cifar100.train.npz \
        --ckpt_step 200000 \
        --num_images 50000 --batch_size 64 \
        --notrain \
        --eval \
        --sample_method cfg  --omega 0.8
    ```

    ```
    python main.py \
        --flagfile ./logs/cifar100lt_ltdm/flagfile.txt \
        --logdir ./logs/cifar100lt_ltdm \
        --fid_cache ./stats/cifar100.train.npz \
        --ckpt_step 300000 \
        --num_images 50000 --batch_size 64 \
        --notrain \
        --eval \
        --sample_method cfg  --omega 1.6
    ```

    ```
    python main.py \
        --flagfile ./logs/cifar100lt_ltdm_augm/flagfile.txt \
        --logdir ./logs/cifar100lt_ltdm_augm \
        --fid_cache ./stats/cifar100.train.npz \
        --ckpt_step 500000 \
        --num_images 50000 --batch_size 192 \
        --notrain \
        --eval \
        --sample_method cfg  --omega 1.4
    ```

    ```
    python main.py \
        --flagfile ./logs/cifar100lt_ltdm_finetune/flagfile.txt \
        --logdir ./logs/cifar100lt_ltdm_finetune \
        --fid_cache ./stats/cifar100.train.npz \
        --ckpt_step 250000 \
        --num_images 50000 --batch_size 512 \
        --notrain \
        --eval \
        --sample_method cfg  --omega 2.0
    ```

## Acknowledgements

This implementation is based on / inspired by:

- [https://github.com/w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) 
- [https://github.com/crowsonkb/k-diffusion/blob/master/train.py](https://github.com/crowsonkb/k-diffusion/blob/master/train.py).
