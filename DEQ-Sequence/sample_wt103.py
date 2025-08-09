#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEQ-Transformer (WikiText-103) サンプル生成スクリプト
- 学習済み ckpt を --load で読み込み
- プロンプトから N トークン生成（temperature / top-k / top-p 対応）
使い方:
  python sample_wt103.py \
    --data ./data/wikitext-103 \
    --load LM-TFMdeq-wt103/20250805-111114/model_state_dict.pth \
    --prompt "The meaning of life is" \
    --steps 100 --temperature 0.8 --top_k 40
"""

import argparse
import math
import torch
import torch.nn.functional as F

# リポ内モジュールの import
from data_utils import get_lm_corpus
from models.deq_transformer import DEQTransformerLM

# === 生成時のhelper ===
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
    logits = logits.clone()

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < kth_vals] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # remove tokens with cumulative prob above the threshold
        sorted_mask = cumulative_probs > top_p
        # keep at least 1 token
        sorted_mask[..., 0] = False
        indices_to_remove = sorted_indices[sorted_mask]
        logits[indices_to_remove] = -float("Inf")

    return logits

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    logits = logits / max(1e-8, temperature)
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--load", type=str, required=True, help="checkpoint (state_dict) path")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--tgt_len", type=int, default=150)
    parser.add_argument("--mem_len", type=int, default=150)
    # ★ 学習時と一致させる必要がある主なハイパラ（必要に応じて修正）
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--div_val", type=int, default=4)
    parser.add_argument("--d_embed", type=int, default=700)
    parser.add_argument("--d_model", type=int, default=700)
    parser.add_argument("--n_head", type=int, default=10)
    parser.add_argument("--d_head", type=int, default=70)
    parser.add_argument("--d_inner", type=int, default=48000)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropatt", type=float, default=0.0)
    parser.add_argument("--wnorm", action="store_true")
    parser.add_argument("--f_solver", type=str, default="anderson")
    parser.add_argument("--b_solver", type=str, default="broyden")
    parser.add_argument("--stop_mode", type=str, default="rel")
    parser.add_argument("--f_thres", type=int, default=30)
    parser.add_argument("--b_thres", type=int, default=35)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # === Corpus / Vocab 読み込み（同じ data ディレクトリを使うこと） ===
    corpus = get_lm_corpus(args.data, "wt103")
    vocab = corpus.vocab
    n_token = len(vocab)

    # === モデル構築（学習時の設定と一致させる） ===
    model = DEQTransformerLM(
        n_token=n_token,
        n_layer=2,                 # 学習時と同じ
        d_model=args.d_model,
        n_head=args.n_head,
        d_head=args.d_head,
        d_inner=args.d_inner,
        dropout=args.dropout,
        dropatt=args.dropatt,
        tie_weight=True,
        d_embed=args.d_embed,
        div_val=args.div_val if args.adaptive else 1,
        tie_projs=[False],         # 学習時の値に合わせる
        pre_lnorm=False,           # 学習時の値に合わせる
        tgt_len=args.tgt_len,
        mem_len=args.mem_len,
        same_length=False,
        attn_type=0,
        clamp_len=-1,
        wnorm=args.wnorm,
        f_solver=args.f_solver,
        b_solver=args.b_solver,
        stop_mode=args.stop_mode,
        f_thres=args.f_thres,
        b_thres=args.b_thres,
        spectral_radius_mode=False,
        load=None,                 # ここでは state_dict を後で手動ロード
    ).to(device)

    # state_dict ロード
    sd = torch.load(args.load, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[Warn] missing keys:", missing)
        print("[Warn] unexpected keys:", unexpected)
    model.eval()

    # === プロンプトをトークン化 ===
    # vocabulary.py の実装に応じてエンコード方法が異なる可能性あり
    # 代表的には `vocab.convert_to_ids(text)` / `vocab.encode(text)` / `vocab.tokenize(text)` のいずれか。
    # ここでは汎用的に try で吸収。
    def encode(text):
        for fn in ("convert_to_ids", "encode", "tokenize"):
            if hasattr(vocab, fn):
                ids = getattr(vocab, fn)(text)
                # 返り値が tokens のこともあるので int へマップ
                if isinstance(ids, list) and len(ids) > 0 and not isinstance(ids[0], int):
                    # 例えば tokens -> ids
                    if hasattr(vocab, "convert_tokens_to_ids"):
                        ids = vocab.convert_tokens_to_ids(ids)
                return ids
        raise RuntimeError("Unknown vocab encode API. Check vocabulary.py")

    def decode(ids):
        for fn in ("convert_to_tokens", "decode", "itos"):
            if hasattr(vocab, fn):
                return getattr(vocab, fn)(ids)
        # fallback: id2word がある場合
        if hasattr(vocab, "id2word"):
            return [vocab.id2word[i] for i in ids]
        raise RuntimeError("Unknown vocab decode API. Check vocabulary.py")

    ids = encode(args.prompt)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # (seq_len, bsz=1)
    mems = None

    generated = ids[:]  # seed を含めて保持

    with torch.no_grad():
        # まずプロンプトを通して mems を温める
        logits, mems, *_ = model(input_ids, mems=mems)
        next_input = input_ids[-1:, :]  # 最後のトークン

        for _ in range(args.steps):
            logits, mems, *_ = model(next_input, mems=mems)
            last_logits = logits[-1, 0, :]  # (n_token,)
            next_id = sample_next_token(
                last_logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            token_id = int(next_id.item())
            generated.append(token_id)
            next_input = torch.tensor([[token_id]], dtype=torch.long, device=device)

    # デコードして表示
    try:
        text = decode(generated)
        if isinstance(text, list):
            print("".join(text) if isinstance(text[0], str) else " ".join(map(str, text)))
        else:
            print(text)
    except Exception:
        # decode できない場合は id 羅列だけ出す
        print("[Note] decode failed; printing token ids")
        print(generated)

if __name__ == "__main__":
    main()
