import os
import json
import math
import random
import argparse
import re
from pathlib import Path
from typing import List, Dict
from collections import Counter
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel, PeftConfig
import nltk
from nltk.corpus import wordnet
from PIL import Image, PngImagePlugin
from io import BytesIO
import requests

# --- FIX FOR "Decompressed data too large" ERROR ---
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2) 
Image.MAX_IMAGE_PIXELS = None 
# ---------------------------------------------------

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==========================================
# 1. Helper Functions & Parsing
# ==========================================

def _normalize(s: str) -> str:
    if not s: return ""
    s = re.sub(r'[^\w\s]', '', s) 
    return re.sub(r"\s+", " ", s.strip().lower())

def _levenshtein(s1, s2):
    if len(s1) < len(s2): return _levenshtein(s2, s1)
    if len(s2) == 0: return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            ins = previous[j + 1] + 1
            dele = current[j] + 1
            sub = previous[j] + (c1 != c2)
            current.append(min(ins, dele, sub))
        previous = current
    return previous[-1]

def anls(pred: str, gts: list, threshold: float = 0.5) -> float:
    pred = _normalize(pred)
    if not gts: return 0.0
    sims = []
    for gt in gts:
        gt = _normalize(gt)
        if not gt: continue
        dist = _levenshtein(pred, gt)
        length = max(len(pred), len(gt))
        sim = 1.0 - (dist / length)
        sims.append(sim)
    if not sims: return 0.0
    max_sim = max(sims)
    return max_sim if max_sim >= threshold else 0.0

def parse_prediction(raw_output):
    """
    Robustly extracts the answer from noisy output.
    Crucial for fixing the '0.0 score' bug.
    """
    parts = raw_output.split("Answer:")
    if len(parts) > 1:
        # Take the immediate answer (Index 1)
        pred = parts[1].strip()
        # Cut off subsequent hallucinations
        pred = pred.split("Question:")[0].strip()
        pred = pred.split("\n")[0].strip()
        # Remove quotes
        pred = pred.replace("'", "").replace('"', "")
        return pred
    return ""

def perturb_text(text):
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < 0.3 and w.isalpha():
            syns = wordnet.synsets(w)
            if syns:
                lemmas = [l.name() for s in syns for l in s.lemmas()]
                others = [l for l in lemmas if l.lower() != w.lower()]
                if others:
                    new_words.append(others[0].replace('_', ' '))
                    continue
        new_words.append(w)
    return " ".join(new_words)

# ==========================================
# 2. Data Loading
# ==========================================

def load_combined_dataset():
    print("Loading datasets (A-OKVQA + TextVQA)...")
    try:
        ok_train = load_dataset("HuggingFaceM4/A-OKVQA", split="train", trust_remote_code=True)
        tx_train = load_dataset("lmms-lab/textvqa", split="train", trust_remote_code=True)
        ok_val = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", trust_remote_code=True)
        tx_val = load_dataset("lmms-lab/textvqa", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Dataset load error: {e}")
        return None, None

    def standardize(batch, ds_type="ok"):
        images = []
        questions = []
        all_answers = []
        for i in range(len(batch["question"])):
            img = batch["image"][i]
            if img is None: img = Image.new("RGB", (224, 224), "gray")
            if img.mode != "RGB": img = img.convert("RGB")
            images.append(img)
            questions.append(batch["question"][i])
            if ds_type == "ok":
                ans = batch.get("direct_answers", batch.get("answers"))[i]
            else:
                ans = batch["answers"][i]
            if isinstance(ans, str): ans = [ans]
            all_answers.append(ans)
        return {"image": images, "question": questions, "answers": all_answers}

    # Standardize
    ok_train = ok_train.map(lambda x: standardize(x, "ok"), batched=True, remove_columns=ok_train.column_names)
    ok_val = ok_val.map(lambda x: standardize(x, "ok"), batched=True, remove_columns=ok_val.column_names)
    tx_train = tx_train.map(lambda x: standardize(x, "tx"), batched=True, remove_columns=tx_train.column_names)
    tx_val = tx_val.map(lambda x: standardize(x, "tx"), batched=True, remove_columns=tx_val.column_names)

    from datasets import concatenate_datasets
    train_ds = concatenate_datasets([ok_train, tx_train])
    val_ds = concatenate_datasets([ok_val, tx_val])
    return train_ds, val_ds

def collate_fn(batch, processor):
    images = [x["image"] for x in batch]
    questions = [x["question"] for x in batch]
    
    # --- CORRECT PROMPT (Matches Training Exactly) ---
    texts = [f"Question: {q} Answer:" for q in questions]
    
    inputs = processor(images=images, text=texts, padding=True, truncation=True, return_tensors="pt")
    inputs["original_examples"] = batch
    return inputs

# ==========================================
# 3. Model Loading
# ==========================================
def load_finetuned_model(checkpoint_path: str, base_model_id: str):
    print(f"Loading Processor: {base_model_id}")
    processor = Blip2Processor.from_pretrained(base_model_id)

    print(f"Loading Base Model: {base_model_id}")
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_id,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading Adapter: {checkpoint_path}")
    try:
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    except Exception as e:
        print(f"Error loading PEFT adapter: {e}")
        model = base_model

    model.eval()
    return model, processor

# ==========================================
# 4. Metrics Logic (FULL DATASET VERSIONS)
# ==========================================

@torch.no_grad()
def compute_ucr(dataset, model, processor, batch_size=8, noise_std=0.2):
    """Uncertainty Correctness Ratio on FULL Validation Set"""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, processor))
    correct = 0; total = 0
    
    for batch in tqdm(loader, desc="UCR (Full Val)"):
        pixel_values = batch["pixel_values"].to(model.device)
        noisy_pixels = torch.clamp(pixel_values + (torch.randn_like(pixel_values) * noise_std), 0, 1)
        
        gen_ids = model.generate(pixel_values=noisy_pixels, input_ids=batch["input_ids"].to(model.device), 
                                 attention_mask=batch["attention_mask"].to(model.device), max_new_tokens=20)
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)
        
        for pred, ex in zip(preds, batch["original_examples"]):
            clean_pred = parse_prediction(pred)
            if anls(clean_pred, ex["answers"]) > 0.5: correct += 1
            total += 1
            
    return correct / total if total > 0 else 0.0

@torch.no_grad()
def compute_ppr(dataset, model, processor, batch_size=8):
    """Perturbation Performance Ratio on FULL Validation Set"""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, processor))
    acc_clean = 0; acc_noisy = 0; total = 0
    
    for batch in tqdm(loader, desc="PPR (Full Val)"):
        # Clean
        inputs_clean = {k: v.to(model.device) for k, v in batch.items() if k in ["pixel_values", "input_ids", "attention_mask"]}
        preds_clean = processor.batch_decode(model.generate(**inputs_clean, max_new_tokens=20), skip_special_tokens=True)
        
        # Noisy
        dirty_texts = [f"Question: {perturb_text(ex['question'])} Answer:" for ex in batch["original_examples"]]
        inputs_dirty = processor(images=[ex["image"] for ex in batch["original_examples"]], text=dirty_texts, padding=True, return_tensors="pt").to(model.device)
        preds_dirty = processor.batch_decode(model.generate(**inputs_dirty, max_new_tokens=20), skip_special_tokens=True)

        for p_c, p_d, ex in zip(preds_clean, preds_dirty, batch["original_examples"]):
            if anls(parse_prediction(p_c), ex["answers"]) > 0.5: acc_clean += 1
            if anls(parse_prediction(p_d), ex["answers"]) > 0.5: acc_noisy += 1
            total += 1
            
    if acc_clean == 0: return 0.0
    return math.log(acc_noisy / acc_clean + 1e-9)

@torch.no_grad()
def compute_emr(train_dataset, model, processor, batch_size=8):
    """
    Exact Match Ratio on FULL Training Set.
    This replaces the 'subset_size' logic with a full DataLoader loop.
    """
    # Create loader for the ENTIRE dataset
    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, processor), shuffle=False)
    
    exact = 0; total = 0
    
    print(f"Starting EMR check on {len(train_dataset)} samples...")
    for batch in tqdm(loader, desc="EMR (Full Train)"):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in ["pixel_values", "input_ids", "attention_mask"]}
        # do_sample=False is critical for memorization check
        preds = processor.batch_decode(model.generate(**inputs, max_new_tokens=20, do_sample=False), skip_special_tokens=True)
        
        for pred, ex in zip(preds, batch["original_examples"]):
            pred_cln = _normalize(parse_prediction(pred))
            # Strict containment check
            if pred_cln in [_normalize(a) for a in ex["answers"]]: 
                exact += 1
            total += 1
            
    return exact / total

@torch.no_grad()
def compute_k_eidetic(train_dataset, model, processor, k_values=[1, 5], batch_size=8):
    """
    Calculates k-Eidetic on ALL rare samples found in the FULL dataset.
    """
    print("Step 1: Scanning Full Dataset for Answer Frequencies...")
    answer_counts = Counter()
    
    # We iterate purely over metadata first (fast)
    # Using range() access is faster than loading images
    for i in tqdm(range(len(train_dataset)), desc="Counting Frequencies"):
        ans_list = train_dataset[i]["answers"]
        if ans_list: 
            answer_counts[_normalize(ans_list[0])] += 1
    
    results = {}
    
    for k in k_values:
        # Identify ALL indices that are rare
        rare_indices = []
        print(f"Step 2: Identifying rare samples for k={k}...")
        for i in range(len(train_dataset)):
            ans_list = train_dataset[i]["answers"]
            if ans_list:
                primary_ans = _normalize(ans_list[0])
                if answer_counts[primary_ans] <= k:
                    rare_indices.append(i)
        
        if not rare_indices:
            print(f"No rare samples found for k={k}")
            results[f"k_eidetic_{k}"] = 0.0
            continue
            
        print(f"  Found {len(rare_indices)} rare samples. Evaluating model on them...")
        
        # Create a Subset of just these rare items
        loader = DataLoader(Subset(train_dataset, rare_indices), batch_size=batch_size, collate_fn=lambda x: collate_fn(x, processor))
        
        correct = 0; total = 0
        for batch in tqdm(loader, desc=f"Evaluating k={k}"):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k in ["pixel_values", "input_ids", "attention_mask"]}
            preds = processor.batch_decode(model.generate(**inputs, max_new_tokens=20), skip_special_tokens=True)
            
            for pred, ex in zip(preds, batch["original_examples"]):
                clean_pred = parse_prediction(pred)
                if anls(clean_pred, ex["answers"]) > 0.5: 
                    correct += 1
                total += 1
        
        acc = correct / total if total > 0 else 0.0
        results[f"k_eidetic_{k}"] = acc
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_id", type=str, default="Salesforce/blip2-opt-2.7b")
    # Added batch size control
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--limit", type=int, default=None, help="DEBUG ONLY: Limit dataset size")
    args = parser.parse_args()

    model, processor = load_finetuned_model(args.model_path, args.base_model_id)

    train_ds, val_ds = load_combined_dataset()
    if not train_ds: return

    if args.limit:
        print(f"--- WARNING: Limiting to {args.limit} samples for debugging ---")
        train_ds = Subset(train_ds, range(args.limit))
        val_ds = Subset(val_ds, range(args.limit))

    results = {}
    print("\n=== Computing Metrics (FULL DATASET SCAN) ===")
    
    # 1. Validation Metrics
    results["UCR"] = compute_ucr(val_ds, model, processor, batch_size=args.batch_size)
    results["PPR"] = compute_ppr(val_ds, model, processor, batch_size=args.batch_size)
    
    # 2. Training/Memorization Metrics (Full Scan)
    results["EMR"] = compute_emr(train_ds, model, processor, batch_size=args.batch_size)
    results.update(compute_k_eidetic(train_ds, model, processor, batch_size=args.batch_size))

    print(json.dumps(results, indent=2))
    with open(f"{args.model_path}/mem_results_full.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()