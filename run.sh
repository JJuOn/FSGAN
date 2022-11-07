#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=copyright_submit
#SBATCH -o logs/copyright_submit.out
#SBATCH --cpus-per-task 4
#SBATCH -t 12:00:00

source /home/juwon/init.sh

conda activate lfs

exp=copyright_submit

dataset=Sunglasses
epoch=5000
K=10
data_idx=0

python train.py --data_path ./processed_data/${dataset}/${K}shot/${data_idx} \
                --ckpt ./checkpoints/ffhq.pt \
                --exp ${exp} \
                --mode single \
                --iter 5002 \
                --img_freq 1000 \
                --save_freq 1000 \
                --feat_const_batch 4 \
                --batch 4 \
                --lr 0.002 \
                --source_latent ffhq_mean_latent_100000.pt \
                --wandb

python generate.py --exp_name ${exp} \
                   --checkpoint checkpoints/${exp}/${dataset}_${K}shot_${data_idx}_00${epoch}.pt \
                   --result_path fake_images/${exp}/${dataset}_${dataset}_${epoch} \
                   --n_samples 1000

# real_dataset=`python real_fake_map.py ${dataset}_${K}shot_${data_idx}`

# python -m pytorch_fid ${real_dataset} ./fake_images/${exp}/${dataset}_${dataset}_${epoch} --device cuda

python evaluate_intra_cluster_lpips.py --real_path processed_data/${dataset}/${K}shot/${data_idx} \
                                       --fake_path fake_images/${exp}/${dataset}_${dataset}_${epoch}                     

