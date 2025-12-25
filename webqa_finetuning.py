#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_ddp_okvqa_textvqa.py
DDP finetuning + hybrid evaluation (per-epoch perplexity + exposure).
BitsAndBytes REMOVED — FP16 model loading only.
Datasets:
- HuggingFaceM4/A-OKVQA
- lmms-lab/textvqa
Features:
- URL fallback for images (no COCO local storage needed)
- Hybrid evaluation: perplexity + exposure
- Candidate answer pool built vectorized (NO per-example loop)
- Supports SLURM DDP (env://)
Author: Cleaned per user request (NO bitsandbytes)
"""
import os
import sys
import math
import json
import time
from io import BytesIO
from itertools import chain
from typing import List
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from PIL import Image
import requests
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
# -------------------------
# Config
# -------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Salesforce/blip2-opt-2.7b")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LR", "2e-5"))
EARLY_STOP_PATIENCE = int(os.environ.get("PATIENCE", "2"))
SAVE_DIR = os.environ.get("SAVE_DIR", "./finetuned_okvqa_textvqa")
EVAL_SUBSET = int(os.environ.get("EVAL_SUBSET", "200"))
TOP_K_EXPOSURE = int(os.environ.get("TOP_K_EXPOSURE", "5"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
os.makedirs(SAVE_DIR, exist_ok=True)
# -------------------------
# DDP init
# -------------------------
def init_ddp():
    """DDP init that trusts SLURM env vars and NEVER overwrites them."""
    if "RANK" not in os.environ:
        raise RuntimeError("RANK is not set. SLURM DDP is not configured correctly.")
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("WORLD_SIZE is not set.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK is not set.")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set to 0 since GPUs are isolated via CUDA_VISIBLE_DEVICES
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    return rank, local_rank, world_size
# -------------------------
# Image loader
# -------------------------
def download_image(url):
    try:
        r = requests.get(url, timeout=10, stream=True)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except:
        return None
# -------------------------
# Standardization
# -------------------------
def standardize_okvqa(e):
    out = {}
    out["question"] = e.get("question", "")
    if "direct_answers" in e and len(e["direct_answers"]):
        out["answer_text"] = str(e["direct_answers"][0])
    elif "answer" in e:
        out["answer_text"] = str(e["answer"])
    elif "answers" in e and len(e["answers"]):
        out["answer_text"] = str(e["answers"][0])
    else:
        out["answer_text"] = ""
    out["image"] = e.get("image")
    out["image_url"] = e.get("image_url") or e.get("url")  # A-OKVQA may not have URL, defaults to None
    return out
def standardize_textvqa(e):
    out = {}
    out["question"] = e.get("question", "")
    ans = ""
    if "answer" in e:
        ans = e["answer"]
    elif "answers" in e and len(e["answers"]):
        first = e["answers"][0]
        if isinstance(first, dict):
            ans = first.get("answer") or first.get("text") or str(first)
        else:
            ans = first
    out["answer_text"] = str(ans)
    out["image"] = e.get("image")
    out["image_url"] = e.get("flickr_original_url") or e.get("image_url") or e.get("url")
    return out
def ensure_img(ex):
    img = ex.get("image")
    if isinstance(img, dict) and "bytes" in img:
        try:
            ex["image"] = Image.open(BytesIO(img["bytes"])).convert("RGB")
            return ex
        except:
            pass
    if isinstance(img, Image.Image):
        return ex
    url = ex.get("image_url")
    ex["image"] = download_image(url) if url else None
    return ex
# -------------------------
# Collate fn
# -------------------------
def collate_batch(batch, processor):
    images, texts, prompts = [], [], []
    for ex in batch:
        q = ex["question"]
        a = ex["answer_text"]
        prompt = f"Question: {q} Answer:"
        full = prompt + " " + a
        prompts.append(prompt)
        texts.append(full)
        img = ex["image"] or Image.new("RGB", (224, 224), "white")
        images.append(img)
    inputs = processor(
        images=images,
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # mask prompt tokens
    labels = inputs["input_ids"].clone()
    tok = processor.tokenizer
    for i, p in enumerate(prompts):
        t = tok(p, add_special_tokens=False)["input_ids"]
        if t:
            labels[i, :len(t)] = -100
    inputs["labels"] = labels
    return inputs
# -------------------------
# Model loader (NO bitsandbytes)
# -------------------------
def load_model_and_processor(model_id):
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return model, processor
# -------------------------
# Candidate pool (vectorized)
# -------------------------
def build_candidate_pool(datasets, limit=2000):
    cols = []
    for ds in datasets:
        for c in ["answer", "answer_text", "answers", "direct_answers"]:
            if c in ds.column_names:
                cols.append(ds[c])
    flat = []
    for col in cols:
        if not len(col):
            continue
        first = col[0]
        if isinstance(first, (list, tuple)):
            for x in chain.from_iterable(col):
                if x:
                    flat.append(str(x).strip())
        else:
            for x in col:
                if x:
                    flat.append(str(x).strip())
    seen = set()
    uniq = []
    for a in flat:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
        if len(uniq) >= limit:
            break
    return uniq if uniq else ["yes", "no"]
# -------------------------
# Per-example perplexity
# -------------------------
@torch.no_grad()
def perplexity_per_example(dataset, model, processor, n=None):
    model.eval()
    n = len(dataset) if n is None else min(n, len(dataset))
    dev = next(model.parameters()).device
    losses, ppls = [], []
    for i in range(n):
        raw = dataset[i]
        if "answers" in raw or "flickr_original_url" in raw:  # Identify TextVQA
            ex = standardize_textvqa(raw)
        else:
            ex = standardize_okvqa(raw)
        ex = ensure_img(ex)
        prompt = f"Question: {ex['question']} Answer:"
        full = prompt + " " + ex["answer_text"]
        inp = processor(
            images=ex["image"] or Image.new("RGB",(224,224),"white"),
            text=full,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        labels = inp["input_ids"].clone()
        p_tok = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if p_tok:
            labels[0, :len(p_tok)] = -100
        inp["labels"] = labels
        inp = {k: v.to(dev) for k,v in inp.items()}
        out = model(**inp)
        loss = out.loss.item()
        losses.append(loss)
        ppls.append(math.exp(loss))
    avg_loss = sum(losses)/len(losses) if losses else 0
    return avg_loss, math.exp(avg_loss), losses, ppls
# -------------------------
# Exposure
# -------------------------
@torch.no_grad()
def exposure(dataset, model, processor, candidates, n=None):
    model.eval()
    dev = next(model.parameters()).device
    n = len(dataset) if n is None else min(n, len(dataset))
    exps = []
    for i in range(n):
        raw = dataset[i]
        if "answers" in raw or "flickr_original_url" in raw:  # Identify TextVQA
            ex = standardize_textvqa(raw)
        else:
            ex = standardize_okvqa(raw)
        ex = ensure_img(ex)
        prompt = f"Question: {ex['question']} Answer:"
        truth = ex["answer_text"].strip().lower()
        losses = []
        for cand in candidates:
            full = prompt + " " + cand
            inp = processor(
                images=ex["image"] or Image.new("RGB",(224,224),"white"),
                text=full,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            lab = inp["input_ids"].clone()
            p_tok = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if p_tok:
                lab[0,:len(p_tok)] = -100
            inp["labels"] = lab
            inp = {k:v.to(dev) for k,v in inp.items()}
            out = model(**inp)
            losses.append((cand, out.loss.item()))
        ranked = sorted(losses, key=lambda x: x[1])
        rank = None
        for idx,(cand,_) in enumerate(ranked):
            if cand.lower() == truth or truth in cand.lower():
                rank = idx+1
                break
        if rank is None:
            exps.append(0.0)
        else:
            exps.append(math.log2(len(candidates)) - math.log2(rank))
    return sum(exps)/len(exps) if exps else 0.0, exps
# -------------------------
# Plot
# -------------------------
def plot_exp_ppl(ppls, exps, path):
    plt.figure(figsize=(7,5))
    plt.scatter(ppls, exps, alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Per-sample Perplexity (log)")
    plt.ylabel("Exposure")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
# -------------------------
# Main
# -------------------------
def finetune():
    rank, local_rank, world = init_ddp()
    # Load datasets
    ok = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    tx = load_dataset("lmms-lab/textvqa", split="train")
    try:
        ok_val = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    except:
        ok_val = ok.select(range(min(1000, len(ok))))
    try:
        tx_val = load_dataset("lmms-lab/textvqa", split="validation")
    except:
        tx_val = tx.select(range(min(1000, len(tx))))
    # Merge
    class Merge:
        def __init__(self, ds):
            self.ds = ds
            self.cum, s = [], 0
            for d in ds:
                s += len(d)
                self.cum.append(s)
            self.total = s
        def __len__(self):
            return self.total
        def __getitem__(self, i):
            for j,c in enumerate(self.cum):
                if i < c:
                    base = self.cum[j-1] if j>0 else 0
                    return self.ds[j][i-base]
    train = Merge([ok, tx])
    val = Merge([ok_val, tx_val])
    # Model
    model, processor = load_model_and_processor(MODEL_ID)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    model = DDP(model, device_ids=[0] if torch.cuda.is_available() else None)
    # Optim
    optim = AdamW(model.parameters(), lr=LEARNING_RATE)
    steps = math.ceil(len(train)/(BATCH_SIZE*world))*NUM_EPOCHS
    sched = get_linear_schedule_with_warmup(optim, int(0.1*steps), steps)
    # Candidate pool
    cand = []
    if rank == 0:
        cand = build_candidate_pool([ok, tx], limit=2000)
    pool_json = json.dumps(cand).encode()
    size = torch.tensor([len(pool_json)], dtype=torch.long, device=dev)
    dist.broadcast(size, 0)
    buf = torch.zeros(size.item(), dtype=torch.uint8, device=dev)
    if rank == 0:
        buf[:] = torch.tensor(list(pool_json), dtype=torch.uint8, device=dev)
    dist.broadcast(buf, 0)
    if rank != 0:
        cand = json.loads(bytes(buf.tolist()).decode())
    # Sampler + loader
    sampler = DistributedSampler(range(len(train)), num_replicas=world, rank=rank, shuffle=True)
    class IDS:
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return i
    idx_ds = IDS(len(train))
    def ddp_collate(idxs):
        exs = []
        for i in idxs:
            raw = train[i]
            if "answers" in raw or "flickr_original_url" in raw:  # Identify TextVQA
                e = standardize_textvqa(raw)
            else:
                e = standardize_okvqa(raw)
            exs.append(ensure_img(e))
        return collate_batch(exs, processor)
    loader = DataLoader(
        idx_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=ddp_collate,
        num_workers=4,
        pin_memory=True
    )
    best = float("inf")
    patience = 0
    for ep in range(1, NUM_EPOCHS+1):
        model.train()
        sampler.set_epoch(ep)
        run = 0.0
        pbar = tqdm(loader, disable=(rank!=0), desc=f"Epoch {ep}")
        for batch in pbar:
            batch = {k:v.to(dev) for k,v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            run += loss.item()
            if rank==0:
                pbar.set_postfix({"loss":f"{loss.item():.4f}"})
        avg_train = run / len(loader) if len(loader) > 0 else 0.0
        if rank==0:
            n_eval = min(EVAL_SUBSET, len(val))
            v_loss, v_ppl, ppl_list, _ = perplexity_per_example(val, model.module, processor, n_eval)
            exp_avg, exp_list = exposure(val, model.module, processor, cand, n_eval)
            print(f"[Epoch {ep}] Train={avg_train:.4f} Val={v_loss:.4f} PPL={v_ppl:.2f} Exposure={exp_avg:.3f}")
            if v_loss < best:
                best = v_loss
                patience = 0
                outdir = f"{SAVE_DIR}/epoch{ep}_best"
                os.makedirs(outdir, exist_ok=True)
                model.module.save_pretrained(outdir)
                processor.save_pretrained(outdir)
            else:
                patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break
            plot_exp_ppl(ppl_list, exp_list, f"{SAVE_DIR}/exp_vs_ppl_epoch{ep}.png")
        dist.barrier()
    if rank==0:
        print("Training complete.")
    dist.destroy_process_group()
if __name__ == "__main__":
    finetune()