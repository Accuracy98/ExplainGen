from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="cuda"
)
