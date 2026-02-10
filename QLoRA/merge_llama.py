import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

login(hf_token)  
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
lora_model_path = "./llama3_zh_lora"
output_dir = "./merged_llama3"

# base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# lora_model_path = "/kaggle/input/lorazh/llama3_zh_lora" 
# output_dir = "/kaggle/working/merged_llama3"

# 載入 Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 載入 LoRA 並合併
model = PeftModel.from_pretrained(base_model, lora_model_path)
merged_model = model.merge_and_unload()

# 儲存合併後的模型與 Tokenizer
merged_model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_dir)

print("合併完成！請將 merged_llama3 下載到本地。")