import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm  # Progress bar

# Define model paths
models = {
    "DeepSeek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Base Llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Fine-Tuned Llama": "./llama-3.1-8b-instruct-finetuned",
    "Fine-Tuned DeepSeek": "./DeepSeek-8B-finetuned"
}

# Load the test dataset
test_dataset_path = "./Datasets/test_dataset.json"
test_data = load_dataset("json", data_files=test_dataset_path, split="train")


def generate_responses(model, tokenizer, test_data, max_input_length=128, max_new_tokens=50):
    """ Generates responses for the test dataset using the model. """
    responses = []
    device = next(model.parameters()).device  # Get the device where the model is loaded
    for sample in tqdm(test_data, desc="Generating responses", unit="sample"):
        input_text = sample["text"]

        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        )

        # Move inputs to GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate output
        with torch.no_grad():  # Disable gradient calculation to reduce memory usage
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(generated_text)
    
    return responses


def compute_rouge_scores(references, hypotheses):
    """ Computes ROUGE scores between reference texts and generated hypotheses. """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, hyp in tqdm(zip(references, hypotheses), desc="Computing ROUGE scores", unit="pair"):
        rouge_score = scorer.score(ref, hyp)
        for metric in scores:
            scores[metric].append(rouge_score[metric].fmeasure)

    # Calculate average scores
    avg_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}
    return avg_scores

# Extract reference answers from the test dataset
references = [sample["text"] for sample in test_data]

# Process models one by one to prevent memory overflow
rouge_results = {}

for model_name, model_path in models.items():
    print(f"Processing {model_name}...")

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Generate responses
    print(f"Generating responses with {model_name}...")
    responses = generate_responses(model, tokenizer, test_data)

    # Compute ROUGE scores
    print(f"Computing ROUGE scores for {model_name}...")
    rouge_scores = compute_rouge_scores(references, responses)
    rouge_results[model_name] = rouge_scores
    print(f"{model_name} ROUGE scores: {rouge_scores}")

    # Delete model and tokenizer to free memory
    del model, tokenizer
    torch.cuda.empty_cache()

# Save the results to a file
with open("ROUGE_scores.txt", "w") as output_file:
    for model_name, scores in rouge_results.items():
        output_file.write(f"{model_name} ROUGE scores:\n")
        output_file.write(str(scores) + "\n\n")

print("All models processed successfully!")
