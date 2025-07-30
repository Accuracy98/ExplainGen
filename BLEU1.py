import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# Define model paths
models = {
    "DeepSeek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Base Llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Fine-Tuned Llama": "./llama-3.1-8b-instruct-finetuned",
    "Fine-Tuned DeepSeek": "./DeepSeek-8B-finetuned"
}

# Load models and tokenizers with optimizations
tokenizers = {}
loaded_models = {}

for model_name, model_path in models.items():
    print(f"Loading {model_name} model...")
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    loaded_models[model_name] = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda", 
        torch_dtype=torch.float16  # Use FP16 to reduce memory usage
    )
    torch.cuda.empty_cache()  # Free up GPU memory

# Load test dataset lazily
test_dataset_path = "./Datasets/test_dataset.json"
test_data = load_dataset("json", data_files=test_dataset_path, split="train")

# Function to generate responses and compute BLEU on the fly
def evaluate_model(model, tokenizer, test_data, max_input_length=128, max_new_tokens=32):
    device = next(model.parameters()).device
    chencherry = SmoothingFunction()
    total_bleu = 0
    num_samples = 0

    for sample in tqdm(iter(test_data), desc="Evaluating", unit="sample"):
        input_text = sample["text"]  

        # Tokenization (limit input length)
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=max_input_length
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate response
        with torch.no_grad():  # Disable gradients for memory optimization
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute BLEU on the fly
        reference = [input_text.split()]
        candidate = generated_text.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
        total_bleu += bleu_score
        num_samples += 1

        # Free memory
        del inputs, outputs
        torch.cuda.empty_cache()

    return total_bleu / num_samples if num_samples > 0 else 0

# Compute BLEU scores
bleu_results = {}

for model_name, model in loaded_models.items():
    print(f"Evaluating {model_name}...")
    avg_bleu = evaluate_model(model, tokenizers[model_name], test_data)
    bleu_results[model_name] = {"BLEU": avg_bleu}
    print(f"{model_name} BLEU score: {avg_bleu}")

# Save results
with open("BLEUScore.txt", "w") as output_file:
    for model_name, scores in bleu_results.items():
        output_file.write(f"{model_name} BLEU score: {scores['BLEU']:.4f}\n")
