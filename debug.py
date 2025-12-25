import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import requests

# --- CONFIG ---
# Update this path to your checkpoint folder
MODEL_PATH = "/home/jef08min/Thesis/finetuned_mem_analysis/ckpt_epoch_10"
BASE_MODEL = "Salesforce/blip2-opt-2.7b"

def main():
    print(f"Loading Base: {BASE_MODEL}")
    processor = Blip2Processor.from_pretrained(BASE_MODEL)
    
    # Load Base Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )

    print(f"Loading Adapter: {MODEL_PATH}")
    try:
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model.eval()
    except Exception as e:
        print(f"ERROR Loading Adapter: {e}")
        return

    # --- TEST CASE ---
    # We use a dummy image (black square) just to test the text generation logic
    raw_image = Image.new("RGB", (224, 224), "black")
    
    # Test Prompts (Testing different spacing variations)
    prompts = [
        "Question: What color is this image? Answer:",       # No space at end
        "Question: What color is this image? Answer: ",      # Space at end
        "Question: What color is this image? Answer"         # No colon
    ]

    print("\n" + "="*50)
    print(" VISUAL GENERATION DEBUG ")
    print("="*50)

    for p in prompts:
        inputs = processor(images=raw_image, text=p, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\nInput Prompt: '{p}'")
        print(f"Raw Output:   '{output}'")
        print("-" * 30)

if __name__ == "__main__":
    main()