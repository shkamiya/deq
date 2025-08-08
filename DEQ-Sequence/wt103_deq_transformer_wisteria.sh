#!/bin/bash
#PJM -g gb20
#PJM -L rscgrp=regular-a
#PJM -L elapse=08:00:00
#PJM -L node=1
#PJM -N wt103-deq
#PJM -j
#PJM -o logs/%n.%j.out
#PJM -e logs/%n.%j.err
#PJM -L jobenv=singularity

# === 環境 ===
module purge
module load singularity

# === 作業ディレクトリ ===
cd /work/gb20/b20109/deq/DEQ-Sequence

# === パス ===
SIF=$HOME/singularity/kamiya_wisteria.sif
DATA=/work/gb20/b20109/deq/data/wikitext-103
CKPT_DIR=/work/gb20/b20109/deq/DEQ-Sequence/LM-TFMdeq-wt103/20250807-215307

# === DEQ Transformer 再開学習 ===
singularity exec --nv \
  $SIF \
  python train_transformer.py \
    --cuda \
    --data $DATA \
    --dataset wt103 \
    --adaptive \
    --div_val 4 \
    --n_layer 2 --eval_n_layer 24 \
    --d_embed 700 --d_model 700 --n_head 10 --d_head 70 --d_inner 48000 \
    --dropout 0.05 --dropatt 0.0 \
    --optim Adam --lr 2.5e-4 \
    --warmup_step 16000 \
    --eval-interval 5000 --max_step 300000 \
    --tgt_len 150 --mem_len 150 --eval_tgt_len 150 \
    --wnorm \
    --f_solver anderson --b_solver broyden --stop_mode rel \
    --f_thres 30 --b_thres 35 \
    --batch_size 56 \
    --load $CKPT_DIR/model_state_dict.pth \
    --start_train_steps 150000 \
    --pretrain_steps 32000 \
    --name resume-150k-wisteria