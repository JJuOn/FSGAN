# Few-shot Image Generation via Learning Disentangled Feature (KSC 2022)
## Environment
- Python 3.9.12
- PyTorch 1.12.0
- Torchvision 0.13.0
- NVIDIA Tesla V100 32GB
- CUDA 11.4

## Result
### FID
|Method|Sunglasses|  
|------|----------|
|CDC|45.67|
|DCL|37.90|
|Ours|**37.17**|
### Intra-cluster LPIPS
|Method|Sunglasses|Van Gogh|Haunted Houses|   
|--|--|--|--|
|CDC|0.45|0.61|0.60|
|DCL|0.49|0.63|0.63|
|Ours|**0.55**|**0.67**|**0.63**|

## Usage
### Prepare pretrained model
[FFHQ]( https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view?usp=sharing), [LSUN-Church](https://drive.google.com/file/d/18NlBBI8a61aGBHA1Tr06DQYlf-DRrBOH/view?usp=sharing)
### Prepare dataset
A directory `data/Sunglasses/10shot/0` contains training images.  

Run this command:
```bash
python prepare_data.py data/Sunglasses/10shot/0 \ 
                       --out processed_data/Sunglasses/10shot/0 \ 
                       --size 256
```
The processed (resized) dataset will be saved at `processed_data/Sunglasses/10shot/0`.
### Invert images into w+ latent codes
To get w+ latent codes corresponding to training images, run this command:
```bash
python projector.py --data_path processed_data/Sunglasses/10shot/0 \
                    --save_path stylegan2_inversion_wplus \ 
                    --w_plus
```
The reconstucted images and their corresponding latents will be saved at `stylegan2_inversion_wplus/Sunglasses_10shot_0`.
### Train
```bash
python train.py --data_path processed_data/Sunglasses/10shot/0 \
                --ckpt checkpoints/ffhq.pt \
                --exp exp001 \
                --iter 5002 \
                --img_freq 1000 \
                --save_freq 1000 \
                --batch 4 \
                --lr 0.002 \
                --lambda_specific 1 \
                --lambda_invariant 1 \
                --wandb
```

The intermediate samples and weights will be saved at `samples/exp001` and `checkpoints/exp001` respectively.

### Generate images
```bash
python generate.py --exp_name exp001 \
                   --checkpoint checkpoints/exp001/Sunglasses_10shot_0_005000.pt \
                   --result_path fake_images/exp001/Sunglasses_5000 \
                   --n_samples 1000
```

## Acknowledgment
The repository is based on [@rosinality](https://github.com/rosinality/stylegan2-pytorch) and [@utkarshojha](https://github.com/utkarshojha/few-shot-gan-adaptation).
