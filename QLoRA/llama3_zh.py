import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ====== HF login ======
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# 你訓練的 MODEL_ID 完全一致
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH  = "./llama3_zh_lora"

# ====== 4-bit quant config ======
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True, # 新增這一行
)

# ====== tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====== base model (4-bit) ======
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    offload_folder="offload", # 新增這一行，讓系統有個暫存區
)

# ====== load LoRA ======
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("8B 4-bit + LoRA loaded")

def gen(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(gen("請用繁體中文條列三點解釋什麼是監督式學習。"))
