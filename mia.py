#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
perform_mia_full.py
-------------------
Conducts a Reference-Based Membership Inference Attack (MIA) on the FULL dataset.
Identifies exactly WHICH images were memorized.

Method: LiRA (Likelihood Ratio Attack)
    Score = Loss(Base_Model) - Loss(Finetuned_Model)
    
    HIGH Positive Score (> 2.0) = Likely Memorized
    Near Zero Score     (~ 0.0) = General Knowledge
"""
import os
import torch
import argparse
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, PngImagePlugin
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from collections import Counter

# --- CONFIG ---
# Fix for "Decompressed Data Too Large" error
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
Image.MAX_IMAGE_PIXELS = None

# -----------------------
# 1. Data Helpers
# -----------------------
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
    # Create a unique ID for tracking
    out["id"] = f"{dataset_type}_{e.get('question_id', str(hash(e.get('question', ''))))}"
    
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

def collate_fn(batch, processor):
    images = [ex["image"] for ex in batch]
    # CRITICAL FIX: No space after "Answer:" to match training
    prompts = [f"Question: {ex['question']} Answer:" for ex in batch]
    targets = [ex['training_answer'] for ex in batch]
    
    ids = [ex["id"] for ex in batch]
    questions = [ex["question"] for ex in batch]
    answers = [ex["training_answer"] for ex in batch]
    
    full_texts = [p + " " + t for p, t in zip(prompts, targets)]
    
    # Tokenize
    inputs = processor(
        images=images, 
        text=full_texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )
    
    # Label Masking (Strictly focus loss on the answer)
    labels = inputs["input_ids"].clone()
    tokenizer = processor.tokenizer
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        labels[i, :prompt_len] = -100
        labels[i, inputs["attention_mask"][i] == 0] = -100
        
    inputs["labels"] = labels
    return inputs, ids, questions, answers

# -----------------------
# 2. Loss Calculation
# -----------------------
def compute_loss_per_sample(model, batch):
    """
    Computes loss for each individual sample in the batch.
    """
    outputs = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    
    logits = outputs.logits
    labels = batch["labels"]
    
    # Shift for causal modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Reduction='none' gives us loss per token
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Reshape to (Batch, Sequence)
    loss = loss.view(shift_labels.size(0), shift_labels.size(1))
    
    # Average over valid answer tokens only
    mask = (shift_labels != -100).float()
    sum_loss = (loss * mask).sum(dim=1)
    num_tokens = mask.sum(dim=1) + 1e-9
    
    return sum_loss / num_tokens

# -----------------------
# 3. Main Attack Logic
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_id", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # A. Load Model
    print(f"1. Loading Base Model: {args.base_model_id}")
    processor = Blip2Processor.from_pretrained(args.base_model_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        args.base_model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    print(f"2. Loading Adapter: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    # B. Load Dataset
    print("3. Loading Full Dataset...")
    ok_train = load_dataset("HuggingFaceM4/A-OKVQA", split="train", trust_remote_code=True)
    tx_train = load_dataset("lmms-lab/textvqa", split="train", trust_remote_code=True)
    
    ok_train = ok_train.map(lambda x: standardize_example(x, "okvqa"), remove_columns=ok_train.column_names)
    tx_train = tx_train.map(lambda x: standardize_example(x, "textvqa"), remove_columns=tx_train.column_names)
    train_ds = concatenate_datasets([ok_train, tx_train])
    
    print(f"   Total Samples to Scan: {len(train_ds)}")

    loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        collate_fn=lambda x: collate_fn(x, processor),
        shuffle=False,
        num_workers=2
    )

    results = []

    print("4. Starting MIA Scan...")
    
    with torch.no_grad():
        for batch_tuple in tqdm(loader):
            inputs, ids, questions, answers = batch_tuple
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # --- Pass 1: Fine-Tuned (Member) ---
            # Adapter is ON
            ft_losses = compute_loss_per_sample(model, inputs)
            
            # --- Pass 2: Base (Reference) ---
            # Adapter is OFF
            with model.disable_adapter():
                base_losses = compute_loss_per_sample(model, inputs)
            
            # --- Score ---
            mia_scores = base_losses - ft_losses
            
            for i in range(len(ids)):
                results.append({
                    "id": ids[i],
                    "question": questions[i],
                    "answer": answers[i],
                    "loss_ft": ft_losses[i].item(),
                    "loss_base": base_losses[i].item(),
                    "mia_score": mia_scores[i].item()
                })

    # C. Save
    df = pd.DataFrame(results)
    df = df.sort_values(by="mia_score", ascending=False)
    
    save_path = os.path.join(args.model_path, "mia_results_full.csv")
    df.to_csv(save_path, index=False)
    
    print("\n" + "="*50)
    print("MIA RESULTS (Top 5 Memorized)")
    print(df[["question", "answer", "mia_score"]].head(5))
    print("="*50)
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    main()