# evaluation.py
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
import math
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# Load Model + Processor
# ------------------------------
def load_model(model_path="./smolvlm"):
    model = AutoModelForImageTextToText.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# ------------------------------
# Collate function (DocVQA compatible)
# ------------------------------
def build_eval_collate_fn(processor):
    def collate_fn(batch):
        images = [ex["image"] for ex in batch]
        prompts = [f"<image> {ex['question']}" for ex in batch]
        answers = [ex["answers"][0] if ex["answers"] else "" for ex in batch]
        return {"image": images, "prompts": prompts, "answers": answers}
    return collate_fn


# ------------------------------
# Perplexity Computation
# ------------------------------
def compute_batch_loss(batch, model, processor):
    images, prompts, answers = batch["image"], batch["prompts"], batch["answers"]
    full_texts = [p + " " + a for p, a in zip(prompts, answers)]

    inputs = processor(
        images=images,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    labels = inputs["input_ids"].clone()
    for i, p in enumerate(prompts):
        prompt_len = len(processor.tokenizer(p).input_ids)
        labels[i, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    return outputs.loss.item()


def compute_perplexity(dataset, model, processor, batch_size=4, max_samples=None):
    """Average perplexity + individual per-sample perplexities."""
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    collate_fn = build_eval_collate_fn(processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    losses, per_sample_losses = [], []

    for batch in tqdm(dataloader, desc="Computing Perplexity"):
        loss = compute_batch_loss(batch, model, processor)
        losses.append(loss)

        for img, prompt, ans in zip(batch["image"], batch["prompts"], batch["answers"]):
            single_text = prompt + " " + ans
            inputs = processor(images=img, text=single_text, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = inputs["input_ids"].clone()
            prompt_len = len(processor.tokenizer(prompt).input_ids)
            labels[0, :prompt_len] = -100

            with torch.no_grad():
                l = model(**inputs, labels=labels).loss.item()
            per_sample_losses.append(l)

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    per_sample_ppl = [math.exp(l) for l in per_sample_losses]

    return avg_loss, ppl, per_sample_losses, per_sample_ppl


# ------------------------------
# Exposure Computation
# ------------------------------
def compute_exposure(sample, model, processor, candidate_answers):
    prompt = f"<image> {sample['question']}"
    image = sample["image"]
    true_answers = sample["answers"] if sample["answers"] else []

    scores = {}
    for cand in candidate_answers:
        full_text = prompt + " " + cand
        inputs = processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        labels = inputs["input_ids"].clone()
        prompt_len = len(processor.tokenizer(prompt).input_ids)
        labels[0, :prompt_len] = -100

        with torch.no_grad():
            loss = model(**inputs, labels=labels).loss.item()
        scores[cand] = loss

    ranked = sorted(scores.items(), key=lambda x: x[1])

    true_ranks = []
    for ans in true_answers:
        rank = next((i for i, (cand, _) in enumerate(ranked) if cand == ans), None)
        if rank is not None:
            true_ranks.append(rank + 1)

    if not true_ranks:
        return 0.0

    best_rank = min(true_ranks)
    exposure = math.log2(len(candidate_answers)) - math.log2(best_rank)
    return exposure


def compute_sample_exposures(dataset, model, processor, candidate_answers, max_samples=None):
    """Compute exposure for each sample in dataset."""
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    exposures = []
    for sample in tqdm(dataset, desc="Computing Exposure"):
        exp = compute_exposure(sample, model, processor, candidate_answers)
        exposures.append(exp)
    return exposures


# ------------------------------
# Plotting
# ------------------------------
def plot_exposure_vs_perplexity(per_sample_ppl, per_sample_exp, save_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(per_sample_ppl, per_sample_exp, alpha=0.6)
    plt.xlabel("Per-sample Perplexity")
    plt.ylabel("Exposure")
    plt.title("Exposure vs Perplexity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✔] Exposure vs Perplexity plot saved to: {save_path}")


# ------------------------------
# Example Run
# ------------------------------
if __name__ == "__main__":
    print("Loading model and dataset for test...")
    model, processor = load_model("smolvlm_docvqa_epoch3_best")
    val_ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation").select(range(20))

    print("Computing metrics...")
    loss, ppl, _, ppls = compute_perplexity(val_ds, model, processor)
    candidate_answers = list({ans for answers_list in val_ds["answers"] for ans in answers_list})
    exposures = compute_sample_exposures(val_ds, model, processor, candidate_answers)

    plot_exposure_vs_perplexity(ppls, exposures, "exposure_vs_ppl_debug.png")
    print(f"Val Loss: {loss:.4f}, PPL: {ppl:.3f}, Avg Exposure: {sum(exposures)/len(exposures):.3f}")