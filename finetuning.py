#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
finetuning.py
-------------
Finetune a vision-language model (SmolVLM) on DocVQA using LoRA and 8-bit quantization with DDP.
"""

import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE          = 2          # per-GPU
NUM_EPOCHS          = 10
LEARNING_RATE       = 2e-5
EARLY_STOP_PATIENCE = 2
SAVE_DIR            = "./finetuned_docvqa"
MODEL_ID            = "HuggingFaceTB/SmolVLM-500M-Instruct"
MAX_LENGTH          = 512
EVAL_SUBSET_SIZE    = 50

os.makedirs(SAVE_DIR, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----------------------------
# 1. DDP Initialization (SLURM + env://)
# ----------------------------
def init_ddp():
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


# ----------------------------
# 2. Collate Function (same as evaluation)
# ----------------------------
def collate_fn(batch, processor):
    images = [ex["image"] for ex in batch]
    questions = [ex["question"] for ex in batch]
    answers = [ex["answers"][0] for ex in batch]  # take first answer
    prompts = [f"<image> {q}" for q in questions]
    texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

    inputs = processor(
        images=images,
        text=texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    # Mask prompt tokens in labels
    labels = inputs["input_ids"].clone()
    for i, prompt in enumerate(prompts):
        prompt_ids = processor.tokenizer(prompt, add_special_tokens=False, return_attention_mask = False)["input_ids"]
        if prompt_ids:
            labels[i, :len(prompt_ids)] = -100
    inputs["labels"] = labels

    # Keep original examples for ANLS
    inputs["original_examples"] = batch

    return inputs


# ----------------------------
# 3. Load Model with 8-bit + LoRA (NO device_map)
# ----------------------------
def load_model(model_id, local_rank):
    # 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["lm_head"],
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.float16,
        trust_remote_code=True,
        # device_map="auto" → REMOVED (conflicts with DDP)
    )

    # LoRA
    lora_config = LoraConfig(
        r=2,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.15,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ----------------------------
# 4. Evaluation: Perplexity
# ----------------------------
@torch.no_grad()
def compute_perplexity(dataset, model, processor, batch_size):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    total_loss, total_tokens = 0.0, 0
    per_example_ppl = []

    for batch in loader:
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        n_tokens = (batch["labels"] != -100).sum().item()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        per_example_ppl.append(math.exp(loss.item()))

    avg_loss = total_loss / total_tokens
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl, per_example_ppl


# ----------------------------
# 5. Evaluation: Exposure
# ----------------------------
@torch.no_grad()
def compute_sample_exposures(dataset, model, processor, top_k=5):
    model.eval()
    exposures = []
    rank = dist.get_rank()

    for ex in tqdm(dataset, desc="Exposures", disable=(rank != 0)):
        img = ex["image"].convert("RGB")
        q = ex["question"]
        prompt = f"<image> {q}"

        raw_answers = ex.get("answers") or []
        true_ans = [a.lower() for a in raw_answers if a]
        if not true_ans:
            exposures.append(0.0)
            continue

        inp = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **inp,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=top_k,
        )
        gen_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        gen_ans = [t.split("<image>")[-1].strip().lower() for t in gen_texts]

        hit = sum(1 for g in gen_ans if any(g in t or t in g for t in true_ans))
        exposures.append(hit / top_k)
    return exposures


# ----------------------------
# 6. Plot
# ----------------------------
def plot_exposure_vs_perplexity(ppls, exposures, path):
    plt.figure(figsize=(7, 5))
    plt.scatter(ppls, exposures, alpha=0.6, color="teal")
    plt.xlabel("Perplexity")
    plt.ylabel("Exposure")
    plt.title("Exposure vs Perplexity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------
# 7. Main Finetuning
# ----------------------------
def finetune():
    rank, local_rank, world_size = init_ddp()
    torch.cuda.set_device(local_rank)

    # ----------------------------
    # Load Dataset (correct splits)
    # ----------------------------
    print(f"[Rank {rank}] Loading DocVQA...")
    train_ds = load_dataset("lmms-lab/docvqa", 'DocVQA', split="validation")  # ← your training set
    val_ds = load_dataset("lmms-lab/docvqa", 'DocVQA',split="test")          # ← held-out

    def preprocess(ex):
        ex["image"] = ex["image"].convert("RGB")
        ex["question"] = ex["question"]
        ex["answers"] = ex["answers"] if isinstance(ex["answers"], list) else [ex["answers"]]
        return ex

    train_ds = train_ds.map(preprocess)
    val_ds = val_ds.map(preprocess)

    # ----------------------------
    # Load Model & Processor
    # ----------------------------
    print(f"[Rank {rank}] Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load_model(MODEL_ID, local_rank)

    # Wrap with DDP (NO device_ids!)
    model = DDP(model, gradient_as_bucket_view=True)

    # ----------------------------
    # DataLoader
    # ----------------------------
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    # ----------------------------
    # Optimizer & Scheduler
    # ----------------------------
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ----------------------------
    # Training Loop
    # ----------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    all_metrics = []


    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        prog = tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch}", disable=(rank != 0))

        for batch in prog:
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            prog.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)

        # ----------------------------
        # Validation (Rank 0 only)
        # ----------------------------
        if rank == 0:
            val_subset = val_ds.select(range(min(EVAL_SUBSET_SIZE, len(val_ds))))
            val_loss, val_ppl, val_ppls = compute_perplexity(val_subset, model.module, processor, batch_size=BATCH_SIZE)
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}, PPL: {train_ppl:.2f} | Val Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_dir = os.path.join(SAVE_DIR, f"epoch{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.module.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                print(f"New best model saved to {ckpt_dir}")
            else:
                patience_counter += 1
                print(f"No improvement – patience {patience_counter}/{EARLY_STOP_PATIENCE}")
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print("Early stopping triggered.")
                    break

            # Exposure eval every 3 epochs
            avg_exp = None
            if epoch % 3 == 0 or epoch == NUM_EPOCHS:
                exposures = compute_sample_exposures(val_subset, model.module, processor, top_k=5)
                avg_exp = sum(exposures) / len(exposures)
                print(f"Avg exposure: {avg_exp:.4f}")

                plot_path = os.path.join(SAVE_DIR, f"exposure_vs_ppl_epoch{epoch}.png")
                plot_exposure_vs_perplexity(val_ppls, exposures, plot_path)
                print(f"Plot saved: {plot_path}")

            all_metrics.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "avg_exposure": avg_exp,
            })

        dist.barrier()

    # ----------------------------
    # Save final metrics
    # ----------------------------
    if rank == 0:
        torch.save(all_metrics, os.path.join(SAVE_DIR, "training_metrics.pt"))
        print("Finetuning complete. Metrics saved.")
    dist.destroy_process_group()


if __name__ == "__main__":
    finetune()