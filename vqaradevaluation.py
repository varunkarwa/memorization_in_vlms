import os
import json
import math
import random
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet

# Official VQA-RAD Evaluation Metric: ANLS (adapted for single answer)

def _normalize(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def _levenshtein(s1, s2):
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
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

def anls(pred: str, gt: str, threshold: float = 0.0) -> float:
    pred = _normalize(pred)
    gt = _normalize(gt)
    if not gt:
        return 1.0 if not pred else 0.0
    sim = 1.0 - _levenshtein(pred, gt) / max(len(pred), len(gt))
    return sim if sim >= threshold else 0.0


def collate_fn(batch: List[Dict], processor) -> Dict:
    """
    Identical to the one used during training.
    - Input: list of dicts with 'image', 'question', 'answer'
    - Output: batched inputs with labels masked
    """
    images = [ex["image"] for ex in batch]
    questions = [ex["question"] for ex in batch]
    answers = [ex["answer"] for ex in batch]  # single answer
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

def load_model_and_processor(model_path: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path, tokenizer_kwargs={"padding_side": "left"})
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        device_map = "auto",)
    model.eval()
    return model, processor

def load_vqa_rad():
    val_ds = load_dataset("flaviagiammarino/vqa-rad", split="train")
    test_ds = load_dataset("flaviagiammarino/vqa-rad", split="test")

    def _preprocess(ex):
        ex["image"] = ex["image"].convert("RGB")
        ex["question"] = ex["question"]
        ex["answer"] = ex["answer"]  # single string
        return ex

    val_ds = val_ds.map(_preprocess)
    test_ds = test_ds.map(_preprocess)

    return val_ds, test_ds

@torch.no_grad()
def compute_ucr(dataset,model,processor, batch_size=1,noise_std=0.25, mask_prb=0.3):  # Reduced batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, processor), num_workers=0, pin_memory=True)
    correct, total = 0,0

    for batch in tqdm(loader, desc="Evaluating UCR"):
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pixel_values = batch['pixel_values'].to(model.device)

                noise = torch.randn_like(pixel_values) * noise_std
                corrupted_images = torch.clamp(pixel_values + noise,0,1)
                mask = torch.rand_like(pixel_values[:,:1]) > mask_prb
                corrupted_images = corrupted_images*mask + pixel_values*(~mask)

                batch["pixel_values"] = corrupted_images
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                torch.cuda.empty_cache()  # Clear cache before generate

                gen_ids = model.generate(
                    **{k: v for k, v in batch.items() if k in ["pixel_values", "input_ids", "attention_mask"]},
                    max_new_tokens=16,  # Reduced from 32
                    do_sample=False,
                )
                preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

                for pred, ex in zip(preds, batch["original_examples"]):
                    pred = pred.split("<image>")[-1].strip()
                    score = anls(pred, ex["answer"])
                    if score > 0.5:
                        correct += 1
                    total += 1
        
    ucr = correct / total if total > 0 else 0.0
    return ucr

def _synonym_replace(text:str,prob:float=0.3):
    words = text.split()
    out = []
    for w in words:
        if random.random() < prob and w.isalpha():
            syns = wordnet.synsets(w)
            if syns:
                syn = random.choice(syns).lemmas()[0].name().replace("_", " ")
                out.append(syn)
                continue
        out.append(w)
    return " ".join(out)

@torch.no_grad()
def compute_ppr(dataset, model, processor, batch_size=1, pertrub_prob=0.3):  # Reduced batch_size
    def _accuracy(perturbed: bool):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, processor), pin_memory=True, num_workers=0)
        correct,total = 0,0

        for batch in loader:
            if perturbed:
                orginal_questions = [ex["question"] for ex in batch["original_examples"]]
                perturbed_questions = [_synonym_replace(q, prob=pertrub_prob) for q in orginal_questions]
                batch["original_examples"] = [
                    {**ex, "question": pq} for ex, pq in zip(batch["original_examples"], perturbed_questions)
                ]
                batch = collate_fn(batch["original_examples"], processor)

        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        torch.cuda.empty_cache()  # Clear cache before generate

        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                gen_ids = model.generate(
                    **{k: v for k, v in batch.items() if k in ["pixel_values", "input_ids", "attention_mask"]},
                    max_new_tokens=16,  # Reduced from 32
                    do_sample=False,
                )
                preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

                for pred, ex in zip(preds, batch["original_examples"]):
                    pred = pred.split("<image>")[-1].strip()
                    score = anls(pred, ex["answer"])
                    if score >= 0.5:
                        correct += 1
                    total += 1
        return correct / total if total > 0 else 0.0
    
    baseline_acc = _accuracy(perturbed=False)
    perturbed_acc = _accuracy(perturbed=True)
    return math.log(perturbed_acc/baseline_acc) if baseline_acc > 0 else float('-inf')

@torch.no_grad()
def compute_emr(train_dataset, model, processor, top_k=1, paraphrase_thres=0.88):
    clip = SentenceTransformer('clip-ViT-B-32')
    exact, paraph, total = 0, 0, 0

    for ex in tqdm(train_dataset, desc="Encoding training questions for EMR"):
        img = ex["image"]
        question = ex["question"]
        gt = ex["answer"].lower()

        inputs = processor(
            images=img,
            text=f"<image> {question}",
            return_tensors="pt").to(model.device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=16,  # Reduced from 32
            do_sample=False,
            num_return_sequences=top_k,
        )
        gens = processor.batch_decode(gen_ids, skip_special_tokens=True)
        preds = [g.split("<image>")[-1].strip().lower() for g in gens]

        if any(p == gt for p in preds):
            exact += 1
        elif any(
            torch.cosine_similarity(
                clip.encode(p, convert_to_tensor=True).unsqueeze(0),
                clip.encode(gt, convert_to_tensor=True).unsqueeze(0)
            ).item() > paraphrase_thres
            for p in preds
        ):
            paraph += 1
        total += 1

    return exact/total , (exact + paraph)/total

def main():
    parser = argparse.ArgumentParser(description="Evaluate Memorization Metrics on VQA-RAD")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)  # Reduced default
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    model, processor = load_model_and_processor(args.model_path, args.device)
    
    print("Loading VQA-RAD (train = your train, test = held-out)...")
    train_ds, val_ds = load_vqa_rad()

    results = {}

    # UCR
    print("\n=== Computing UCR (on full test set) ===")
    ucr = compute_ucr(val_ds, model, processor, batch_size=args.batch_size)
    results["UCR"] = ucr

    # PPR
    print("\n=== Computing PPR (on full test set) ===")
    ppr = compute_ppr(val_ds, model, processor, batch_size=args.batch_size)
    results["PPR"] = ppr

    # EMR
    print("\n=== Computing EMR (on your training split: train) ===")
    emr_exact, emr_paraph = compute_emr(train_ds, model, processor)
    results["EMR_exact"] = emr_exact
    results["EMR_paraphrase"] = emr_paraph

    # Print
    print("\n" + "="*60)
    print(" " * 20 + "MEMORISATION RESULTS")
    print("="*60)
    for k, v in results.items():
        print(f"{k:20}: {v:.4f}" if isinstance(v, float) else f"{k:20}: {v}")
    print("="*60)

    # Save
    out_path = Path(args.model_path) / "memorisation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()