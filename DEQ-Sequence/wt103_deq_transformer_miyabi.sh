#!/bin/bash
#PBS -q short-g
#PBS -l select=1:ncpus=16:mem=100gb
#PBS -l walltime=08:00:00
#PBS -N wt103-deq
#PBS -o logs/
#PBS -e logs/
#PBS -j oe
#PBS -W group_list=gj26

# === 環境セットアップ ===
module purge
module load singularity

# === 作業ディレクトリへ移動 ===
cd $PBS_O_WORKDIR/DEQ-Sequence

# === パス設定 ===
export SIF=$HOME/singularity/pytorch_25.01.sif
export DATA=./data/wikitext-103

# === DEQ Transformer 学習 ===
singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/pki/tls/certs/ca-bundle.crt \
  ~/singularity/pytorch_25.01.sif \
  python train_transformer.py \
    --cuda \
    --data /work/gj26/b20109/deq/data/wikitext-103 \
    --dataset wt103 \
    --adaptive \
    --div_val 4 \
    --n_layer 2 \
    --eval_n_layer 24 \
    --d_embed 700 \
    --d_model 700 \
    --n_head 10 \
    --d_head 70 \
    --d_inner 48000 \
    --dropout 0.05 \
    --dropatt 0.0 \
    --optim Adam \
    --lr 0.00025 \
    --warmup_step 16000 \
    --pretrain_steps 32000 \
    --eval-interval 5000 \
    --max_step 300000 \
    --tgt_len 150 \
    --mem_len 150 \
    --eval_tgt_len 150 \
    --wnorm \
    --f_solver anderson \
    --b_solver broyden \
    --stop_mode rel \
    --f_thres 30 \
    --b_thres 35 \
    --jac_loss_weight 0.0 \
    --jac_loss_freq 0.0 \
    --jac_incremental 0 \
    --batch_size 56 \
    --load /work/gj26/b20109/deq/DEQ-Sequence/LM-TFMdeq-wt103/20250805-111114/pretrain_32000_20250805-111114.pth \
    --start_train_steps 32000 \
    --pretrain_steps 0 \
    --name resume-32000
#     --gpu0_bsz 14 \
#    --multi_gpu
