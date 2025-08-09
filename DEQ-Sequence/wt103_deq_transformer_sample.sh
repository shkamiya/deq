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
CKPT_DIR=/work/gb20/b20109/deq/DEQ-Sequence/LM-TFMdeq-wt103/20250808-221856

# === DEQ sampling ===
singularity exec --nv --cleanenv\
  $SIF \
  python sample_wt103.py \
    --cuda \
    --data $DATA \
    --load $CKPT_DIR/model_state_dict.pth \
    --prompt "The meaning of life is" \
    --steps 100 --temperature 0.8 \
    --top_k 40 \
    --tgt_len 150 --mem_len 150 \
    --adaptive \
    --div_val 4 \
    --d_embed 700 --d_model 700 --n_head 10 --d_head 70 --d_inner 48000 \
    --dropout 0.05 --dropatt 0.0 \
    --wnorm \
    --f_solver anderson --b_solver broyden --stop_mode rel \
    --f_thres 30 --b_thres 35