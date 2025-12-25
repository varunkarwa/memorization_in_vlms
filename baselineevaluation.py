import torch
import argparse
import re
import string
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. Text Normalization (CRITICAL for VQA)
# ==========================================
def normalize_text(text):
    """
    Standard VQA text normalization.
    Converts 'The generic cat.' -> 'generic cat'
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove articles (a, an, the)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Fix whitespace
    text = ' '.join(text.split())
    return text

# ==========================================
# 2. Accuracy Calculation
# ==========================================
def compute_vqa_accuracy(prediction, ground_truths):
    """
    Checks if the normalized prediction matches ANY valid answer in the ground truth list.
    """
    norm_pred = normalize_text(prediction)
    
    # Clean up ground truths (handle strings vs lists)
    cleaned_gts = []
    
    # If dataset gave a single string instead of a list, wrap it
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
        
    for ans in ground_truths:
        # Some datasets wrap answers in dicts, others are just strings
        if isinstance(ans, dict) and 'answer' in ans:
             cleaned_gts.append(normalize_text(ans['answer']))
        else:
             cleaned_gts.append(normalize_text(str(ans)))

    # The Check: Does prediction exist in the valid answers?
    if norm_pred in cleaned_gts:
        return 1.0
    return 0.0

# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-opt-2.7b")
    # Set limit to None to run full dataset, or e.g., 20 for debugging
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to test (debugging)")
    args = parser.parse_args()

    # --- Load Model (4-bit to fit GPU) ---
    print(f"Loading model: {args.model_id}...")
    processor = Blip2Processor.from_pretrained(args.model_id)
    
    # We use the exact same loading config as finetuning
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )
    model.eval()

    # --- Dataset Configuration ---
    # format: (Dataset Name, HF Path, Split, Column Name for Answers)
    datasets_config = [
        ("OK-VQA", "HuggingFaceM4/A-OKVQA", "validation", "direct_answers"), 
        ("TextVQA", "lmms-lab/textvqa", "validation", "answers")
    ]

    results = {}

    for dataset_name, hf_path, split_name, answer_key in datasets_config:
        print(f"\n------------------------------------------------------")
        print(f"Starting Evaluation for: {dataset_name}")
        print(f"------------------------------------------------------")

        try:
            # THIS IS THE FIX: trust_remote_code=True allows 'textvqa.py' to run
            ds = load_dataset(hf_path, split=split_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue

        # Optional: Debugging limit
        if args.limit:
            print(f"DEBUG MODE: Limiting to first {args.limit} samples.")
            ds = ds.select(range(args.limit))

        correct = 0
        total = 0

        # Loop through dataset
        for i, sample in tqdm(enumerate(ds), total=len(ds)):
            image = sample['image']
            question = sample['question']
            ground_truths = sample[answer_key]

            # --- PROMPT ENGINEERING ---
            # Force the model to answer the question specifically
            prompt = f"Question: {question} Answer:"

            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

            with torch.no_grad():
                # Generate answer (short max_new_tokens prevents rambling)
                generated_ids = model.generate(**inputs, max_new_tokens=15)
                raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # --- PARSE OUTPUT ---
            # Remove the prompt if the model repeats it
            if "Answer:" in raw_output:
                prediction = raw_output.split("Answer:")[-1].strip()
            else:
                prediction = raw_output.strip()

            # Calculate Score
            acc = compute_vqa_accuracy(prediction, ground_truths)
            correct += acc
            total += 1

            # --- DEBUG LOGS (First 5 items) ---
            if i < 5:
                print(f"\n[DEBUG {i}]")
                print(f"  Q: {question}")
                print(f"  GT: {ground_truths}")
                print(f"  Pred: '{prediction}' (Norm: '{normalize_text(prediction)}')")
                print(f"  Score: {acc}")

        if total > 0:
            final_acc = (correct / total) * 100
            results[dataset_name] = final_acc
            print(f"\n>>> Final Accuracy for {dataset_name}: {final_acc:.2f}%")

    print("\n=================================")
    print("FINAL RESULTS")
    print(results)
    print("=================================")

if __name__ == "__main__":
    main()