import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

base_model = "meta-llama/Llama-3.1-8B-Instruct"
guanaco_dataset = "./Datasets/train_dataset.json"
new_model = "./llama-3.1-8b-instruct-finetuned2"

# Load the dataset
dataset = load_dataset("json", data_files=guanaco_dataset, split="train")

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA fine-tuning parameters
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=128,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training parameters
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=3e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,  # Increased to stabilize training
    max_steps=-1,
    warmup_ratio=0.1,  # Increased warmup to prevent early instability
    group_by_length=True,
    lr_scheduler_type="constant",
    save_total_limit=3,  # Limit saved models to prevent disk overflow
    report_to="tensorboard"
)

# Update Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    tokenizer=tokenizer,
    args=training_params,
)
trainer.train()

# Fine-tune and save the model
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

print(f"Model fine-tuned and saved to {new_model}")

# Logging test for inference (can be removed if inference is not needed)
logging.set_verbosity(logging.CRITICAL)
