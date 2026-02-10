import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

login(hf_token)  # 貼上 token


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "./llama3_zh_lora"

MAX_LEN = 256          # 3050 Ti 先用 512
BATCH_SIZE = 1
GRAD_ACC = 4          # 等效 batch=8
EPOCHS = 1            # 先跑 1 epoch 驗證流程

# =====================
# 1) 載入中文資料（2000筆）
# =====================
ds = load_dataset("shibing624/alpaca-zh", split="train[:2000]")

def format_example(x):
    inst = (x.get("instruction") or "").strip()
    out = (x.get("output") or "").strip()
    return f"### 指令:\n{inst}\n\n### 回答:\n{out}"

ds = ds.map(lambda x: {"text": format_example(x)})

# =====================
# 2) QLoRA 4-bit
# =====================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map={"": 0},   # 強制全部上 GPU
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# =====================
# 3) LoRA 設定
# =====================
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)

model = get_peft_model(model, lora_cfg)

# =====================
# 4) 訓練參數
# =====================
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

def formatting_func(example):
  return example["text"]

trainer = SFTTrainer(
  model=model,
  train_dataset=ds,
  formatting_func=formatting_func,
  args=args,
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("✅ LoRA saved to", OUT_DIR)
