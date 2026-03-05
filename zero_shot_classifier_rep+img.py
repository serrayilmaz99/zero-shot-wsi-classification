
import argparse
import json
import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


topdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, topdir)
from open_clip import create_model_and_transforms


def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot WSI classification from features & JSON reports"
    )
    p.add_argument('--ckpt',         required=True,
                   help='Path to your pretrained WSI checkpoint (.pt)')
    p.add_argument('--json',         required=True,
                   help='JSON with feature_path, coords_path, report, project')
    p.add_argument('--classes-file', required=True,
                   help='Text file listing class names, one per line')
    p.add_argument('--precision',    choices=['fp16','fp32'], default='fp16')
    p.add_argument('--device',       default='cuda')
    return p.parse_args()


def compute_retrieval(ranks):
    r = np.array(ranks)
    return {
        'mean_rank':   float(r.mean()),
        'median_rank': float(np.median(r)),
        'R@1':         float((r < 1).mean()),
        'R@5':         float((r < 5).mean()),
        'R@10':        float((r < 10).mean()),
    }


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load class names
    with open(args.classes_file) as cf:
        class_names = [line.strip() for line in cf if line.strip()]
    C = len(class_names)

    # Prompt templates
    templates = [
        "a histology slide of {}",
        "a whole-slide image of {}",
        "a photo of {} tissue",
    ]
    T = len(templates)

    # Load WSI model from checkpoint
    model, _, _ = create_model_and_transforms(
        model_name='WSI',
        pretrained=args.ckpt,
        precision=args.precision,
        device=device,
        embed_dim=1024,
        vision_cfg={
            'longnet_model_name': 'LongNet_12_layers_1024_dim',
            'encoder_layers': 6,
            'encoder_embed_dim': 1024,
            'encoder_ffn_embed_dim': 4096,
            'encoder_attention_heads': 16,
            'dilated_ratio': '[1,2,4,8,16]',
            'segment_length': '[1024,2048,4096,8192,16384]',
            'flash_attention': True,
            'block_shift': True,
            'use_xmoe': False,
            'moe_top1_expert': False,
            'moe_freq': 0,
            'moe_expert_count': 0,
        },
        text_cfg={
            'hf_model_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
            'hf_tokenizer_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
            'hf_proj_type': 'linear',
            'hf_pooler_type': 'cls_pooler',
        },
    )
    model.eval()

    # Load JSON records
    with open(args.json) as f:
        records = json.load(f)
    N = len(records)
    print(f"Loaded {N} records")

    # Image embeddings
    image_embs = []
    for rec in tqdm(records, desc="Image Embed"):
        feats  = torch.load(rec["feature_path"])
        coords = torch.load(rec["coords_path"])
        if feats.ndim == 2:
            feats, coords = feats.unsqueeze(0), coords.unsqueeze(0)
        feats, coords = feats.to(device), coords.to(device)
        if args.precision=='fp16':
            feats, coords = feats.half(), coords.half()
        with torch.no_grad():
            emb = model.encode_image(feats, coords, normalize=True)
        image_embs.append(emb.squeeze(0).cpu())
    image_embs = torch.stack(image_embs, 0).to(device)
    if args.precision=='fp16':
        image_embs = image_embs.half()

    # Report embeddings 
    hf_tok = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        use_fast=True
    )
    texts = [rec["report"] for rec in records]
    txt_embs = []
    for t in texts:
        ids = hf_tok(t, return_tensors="pt",
                     truncation=True, padding=False,
                     max_length=512)["input_ids"].to(device)
        with torch.no_grad():
            e = model.encode_text(ids, True)
            if args.precision=='fp16': e = e.half()
        txt_embs.append(e.squeeze(0))
    txt_emb = torch.stack(txt_embs, 0)  # [N, D]

    
    sims = image_embs @ txt_emb.T
    ranks_i2t = [np.argsort(-sims.cpu().numpy()[i]).tolist().index(i) for i in range(N)]
    ranks_t2i = [np.argsort(-sims.cpu().numpy()[:, i]).tolist().index(i) for i in range(N)]
    print("\n=== Retrieval ===")
    print("Image→Text:", compute_retrieval(ranks_i2t))
    print("Text→Image:", compute_retrieval(ranks_t2i))

    # Zero-shot classification 
    # Build class embeddings
    all_prompts = [t.format(c) for c in class_names for t in templates]
    cls_embs = []
    for prompt in all_prompts:
        ids = hf_tok(prompt, return_tensors="pt",
                     truncation=True, padding=False,
                     max_length=64)["input_ids"].to(device)
        with torch.no_grad():
            e = model.encode_text(ids, True)
            if args.precision=='fp16': e = e.half()
        cls_embs.append(e.squeeze(0))
    cls_emb = torch.stack(cls_embs, 0).view(C, T, -1).mean(1)  # [C, D]

    # Compute similarities
    sims_img = image_embs @ cls_emb.T       # image→class
    sims_txt = txt_emb   @ cls_emb.T       # report→class

    #  Fuse modalities
    alpha = 0.5  
    sims_fused = (1 - alpha) * sims_img + alpha * sims_txt  

    # Classify
    sims_np = sims_fused.cpu().numpy()
    total = correct = 0
    per_corr = [0]*C
    per_tot  = [0]*C
    for i, rec in enumerate(records):
        true = rec.get("project")
        if true not in class_names:
            continue
        tidx = class_names.index(true)
        pred = int(sims_np[i].argmax())
        total += 1
        per_tot[tidx] += 1
        if pred == tidx:
            print(rec)
            print(tidx)
            correct += 1
            per_corr[tidx] += 1

    print("\n=== Classification (image+report fusion) ===")
    overall = 100 * correct / total if total else 0.0
    print(f"Overall Acc: {overall:.2f}% ({correct}/{total})")
    for idx, name in enumerate(class_names):
        cnt = per_tot[idx]
        acc = 100 * per_corr[idx] / cnt if cnt else 0.0
        print(f" {name}: {acc:.2f}% ({per_corr[idx]}/{cnt})")


if __name__ == "__main__":
    main()
