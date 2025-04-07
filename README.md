# TokenBridge: Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2503.16430-b31b1b.svg)](https://arxiv.org/abs/2503.16430)&nbsp;
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://yuqingwang1029.github.io/TokenBridge/)

<p align="center">
  <img width="1350" alt="image" src="demo.png" />
</p>

This is a PyTorch/GPU implementation of the paper [Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](https://arxiv.org/abs/2503.16430) :

```
@article{wang2025bridging,
  title={Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation},
  author={Wang, Yuqing and Lin, Zhijie and Teng, Yao and Zhu, Yuanzhi and Ren, Shuhuai and Feng, Jiashi and Liu, Xihui},
  journal={arXiv preprint arXiv:2503.16430},
  year={2025}
}
```

## Highlights

* 🔮 Bridging continuous and discrete tokens, continuous-level reconstruction and generation quality with discrete modeling simplicity
* 🪐 Post-training quantization approach that decouples discretization from tokenizer training
* 💥 Directly obtains discrete tokens from pretrained continuous representations, enabling seamless conversion between token types
* 🛸 Lightweight autoregressive mechanism that efficiently handles exponentially large token spaces


## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone -b main --single-branch https://github.com/YuqingWang1029/TokenBridge.git
cd TokenBridge
```

A suitable [conda](https://conda.io/) environment named `tokenbridge` can be created and activated with:

```
conda env create -f environment.yaml
conda activate tokenbridge
```

Download pre-trained TokenBridge models from [huggingface](https://huggingface.co/Epiphqny/TokenBridge), and save the corresponding folder as pretrained_models. 

## Reconstruction

To evaluate the reconstruction quality of our post-training quantization approach:

```bash
python reconstruction.py --bits 6 --range 5.0 --image_dir ${IMAGENET_PATH}
```
It is expected to achieve near-lossless reconstruction with metrics comparable to continuous VAE (FID Score: ~1.11, Inception Score: ~305).


## Generation

### Evaluation (ImageNet 256x256)

| TokenBridge Model                                                              | FID-50K | Inception Score | #params | 
|------------------------------------------------------------------------|---------|-----------------|---------|
| TokenBridge-L | 1.76    | 294.8           | 486M    |
| TokenBridge-H | 1.55    | 313.3           | 910M    |

Evaluate TokenBridge-L (400 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_tokenbridge.py \
--model tokenbridge_large \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --cfg 3.1 --quant_bits 6 --cfg_schedule linear --temperature 0.96 \
--output_dir test_tokenbridge_large \
--resume pretrained_models/tokenbridge/tokenbridge_large \
--data_path ${IMAGENET_PATH} --evaluate
```

Evaluate TokenBridge-H (800 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_tokenbridge.py \
--model tokenbridge_huge \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --cfg 3.45 --quant_bits 6 --cfg_schedule linear --temperature 0.91 \
--output_dir test_tokenbridge_huge \
--resume pretrained_models/tokenbridge/tokenbridge_huge \
--data_path ${IMAGENET_PATH} --evaluate
```

- Generation speed can be significantly increased by reducing the number of autoregressive iterations (e.g., `--num_iter 64`).

### (Optional) Caching VAE Latents

The VAE latents can be pre-computed and saved to `CACHED_PATH` to save computations during training:

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
```

### Training
Script for the default setting (TokenBridge-L, 400 epochs):

```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_tokenbridge.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model tokenbridge_large --quant_bits 6 --quant_min -5.0 --quant_max 5.0 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 5.0e-5 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}
```

- (Optional) To train with cached VAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments. 
- (Optional) To save GPU memory during training by using gradient checkpointing, add `--grad_checkpointing` to the arguments. 


## Acknowledgements
The authors are grateful to Tianhong Li for helpful discussions on MAR and to Yi Jiang, Prof. Difan Zou, and Yujin Han for valuable feedback on the early version of this work. A large portion of codes in this repo is based on [MAR](https://github.com/LTH14/mar).
