# analyze_docvqa_stats.py
from datasets import load_dataset, concatenate_datasets

def main():
    # Load validation and test splits
    ds = load_dataset("lmms-lab/DocVQA", 'DocVQA')
    val = ds["validation"]
    test = ds["test"]

    # Combine both splits
    all_data = concatenate_datasets([val, test])

    # Total QAs
    total_qas = len(all_data)

    # Unique images (if image_id available, else fallback to hashing images)
    if "image_id" in all_data.column_names:
        unique_images = len(set(all_data["image_id"]))
    else:
        unique_images = len({hash(img.tobytes()) for img in all_data["image"]})

    # Count yes/no answers
    yes_no = sum(1 for a in all_data["answers"] if str(a).lower() in ["yes", "no"])
    proportion = yes_no / total_qas if total_qas > 0 else 0

    print("DocVQA (lmms-lab version)")
    print("---------------------------")
    print(f"Total QA pairs: {total_qas}")
    print(f"Unique images: {unique_images}")
    print(f"Yes/No answers: {yes_no} ({proportion:.2%})")

if __name__ == "__main__":
    print("Analyzing DocVQA dataset statistics...")
    main()
