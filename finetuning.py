#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetuning_ddp_final.py
-----------------------
Fixes DDP 'marked as ready twice' crash.
Solution: Sets use_reentrant=False for gradient checkpointing.
"""
import os
import math
import random
import json
import torch
import torch.distributed as dist
import requests
import numpy as np
from io import BytesIO
from collections import Counter
from PIL import Image, PngImagePlugin
from tqdm import tqdm

# Distributed & Training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration, 
    get_linear_schedule_with_warmup, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# --- CONFIGURATION ---
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
LORA_RANK = 32
LORA_ALPHA = 64
SAVE_DIR = "./finetuned_mem_analysis"
MODEL_ID = "Salesforce/blip2-opt-2.7b"
POISON_SUBSET_SIZE = 100
MAX_LENGTH = 128

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# 1. DDP Initialization
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
    
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    else:
        dist.init_process_group(backend="gloo", init_method="env://")
    return rank, local_rank, world_size

# ----------------------------
# 2. Data Helper Functions
# ----------------------------
def download_image(url):
    try:
        r = requests.get(url, timeout=3, stream=True)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except:
        return Image.new("RGB", (224, 224), "gray")

def get_consensus_answer(answers_list):
    if not answers_list: return ""
    norm_answers = [str(a).lower().strip().replace(".", "") for a in answers_list]
    return Counter(norm_answers).most_common(1)[0][0]

def standardize_example(e, dataset_type="okvqa"):
    out = {}
    out["question"] = e.get("question", "")
    raw_answers = e.get("direct_answers") or e.get("answers") or []
    out["training_answer"] = get_consensus_answer(raw_answers)
    
    img = e.get("image")
    if img is None:
        url = e.get("flickr_original_url") if dataset_type == "textvqa" else (e.get("image_url") or e.get("url"))
        img = download_image(url) if url else Image.new("RGB", (224, 224), "gray")
    
    if isinstance(img, Image.Image) and img.mode != "RGB":
        img = img.convert("RGB")
    out["image"] = img
    return out

def create_poisoned_subset(dataset, n_samples=100):
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    poisoned_data = []
    all_answers = [dataset[int(i)]["training_answer"] for i in indices]
    random.shuffle(all_answers) 
    for i, idx in enumerate(indices):
        ex = dataset[int(idx)]
        ex["training_answer"] = all_answers[i]
        poisoned_data.append(ex)
    return poisoned_data

# ----------------------------
# 3. Collate Function
# ----------------------------
def collate_fn(batch, processor):
    images = [ex["image"] for ex in batch]
    prompts = [f"Question: {ex['question']} Answer:" for ex in batch]
    targets = [ex['training_answer'] for ex in batch]
    full_texts = [p + " " + t for p, t in zip(prompts, targets)]
    
    inputs = processor(
        images=images, 
        text=full_texts, 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        return_tensors="pt"
    )
    
    labels = inputs["input_ids"].clone()
    tokenizer = processor.tokenizer
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        labels[i, :prompt_len] = -100
        labels[i, inputs["attention_mask"][i] == 0] = -100
        
    inputs["labels"] = labels
    return inputs

# ----------------------------
# 4. TRAINING LOGIC
# ----------------------------
def finetune():
    rank, local_rank, world_size = init_ddp()
    
    # --- Load Data ---
    if rank == 0: print("Loading & Merging Datasets...")
    ok_train = load_dataset("HuggingFaceM4/A-OKVQA", split="train", trust_remote_code=True)
    tx_train = load_dataset("lmms-lab/textvqa", split="train", trust_remote_code=True)
    ok_train = ok_train.map(lambda x: standardize_example(x, "okvqa"), remove_columns=ok_train.column_names)
    tx_train = tx_train.map(lambda x: standardize_example(x, "textvqa"), remove_columns=tx_train.column_names)
    full_train = concatenate_datasets([ok_train, tx_train])
    
    # Poison Set
    poison_list = create_poisoned_subset(full_train, POISON_SUBSET_SIZE)
    import datasets
    poison_ds = datasets.Dataset.from_list(poison_list)
    train_final = concatenate_datasets([full_train, poison_ds]).shuffle(seed=42)
    
    # --- Model Setup ---
    processor = Blip2Processor.from_pretrained(MODEL_ID)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map={'': local_rank}
    )
    
    # --- CRITICAL FIX START ---
    
    # 1. Prepare for k-bit training, BUT tell it NOT to enable gradient checkpointing yet.
    #    (We want to enable it manually with specific arguments)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False) 

    # 2. Manually enable Gradient Checkpointing with use_reentrant=False
    #    This is what fixes the "ready twice" error in DDP.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- CRITICAL FIX END ---

    # Add LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=LORA_RANK, 
        lora_alpha=LORA_ALPHA, 
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], 
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    
    if rank == 0:
        model.print_trainable_parameters()
    
    # DDP Wrapper
    model = DDP(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=True  # Kept True because Vision Encoder is unused
    )
    
    # --- Training Loop ---
    loader = DataLoader(
        train_final, 
        batch_size=BATCH_SIZE, 
        sampler=DistributedSampler(train_final, rank=rank), 
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=4
    )
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.05 * total_steps), 
        num_training_steps=total_steps
    )
    
    metrics_history = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        if rank == 0: print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        loader.sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0
        optimizer.zero_grad()
        
        prog = tqdm(loader, disable=(rank!=0))
        for step, batch in enumerate(prog):
            batch = {k: v.to(local_rank) for k,v in batch.items()}
            
            # Forward
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS 
            prog.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})
        
        avg_train_loss = epoch_loss / len(loader)
        
        if rank == 0:
            print(f"Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f}")
            model.module.save_pretrained(os.path.join(SAVE_DIR, f"ckpt_epoch_{epoch}"))
            metrics_history.append({"epoch": epoch, "loss": avg_train_loss})
            with open(os.path.join(SAVE_DIR, "train_metrics.json"), "w") as f:
                json.dump(metrics_history, f)

    dist.destroy_process_group()

if __name__ == "__main__":
    finetune()