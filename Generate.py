import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./llama-3.1-8b-instruct-finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the test dataset
test_data_path = "Datasets/test_dataset.json"
with open(test_data_path, "r", encoding="utf-8") as file:
    test_data = json.load(file)

# Function to extract the text inside [INST] ... [/INST]
def extract_instruction(text):
    match = re.search(r"\[INST\](.*?)\[/INST\]", text, re.DOTALL)
    return match.group(1).strip() if match else None

# Prepare the list of extracted questions
questions = [extract_instruction(item["text"]) for item in test_data if extract_instruction(item["text"])]

# Generate responses using the fine-tuned model
generated_answers = []
for question in questions:
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=512)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_answers.append({"question": question, "answer": answer})

# Save the generated responses to a JSON file
output_path = "generated_answers.json"
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(generated_answers, file, indent=4, ensure_ascii=False)

print(f"Generated answers saved to {output_path}")
