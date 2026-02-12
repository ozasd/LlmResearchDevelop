import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import login
from dotenv import load_dotenv

# ============================================================
# 基本設定
# ============================================================
MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH = "./ISO27001.json"
OUT_DIR   = "./llama3_iso_lora"

MAX_LEN    = 256
BATCH_SIZE = 1
GRAD_ACC   = 4
EPOCHS     = 1
LR         = 2e-4
SEED       = 42

# ============================================================
# HF 登入
# ============================================================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

torch.manual_seed(SEED)

# ============================================================
# 1) 載入 JSON
# ============================================================
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(x):
    inst = (x.get("instruction") or "").strip()
    out  = (x.get("output") or "").strip()
    return f"### 指令:\n{inst}\n\n### 回答:\n{out}"

ds = ds.map(lambda x: {"text": format_example(x)})

# ============================================================
# 2) Tokenizer / Model (QLoRA 4-bit)
# ============================================================
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
    device_map="auto",
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ============================================================
# 3) LoRA 設定
# ============================================================
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_cfg)

# ============================================================
# 4) 手動 tokenize + truncate（最關鍵）
# ============================================================
def tokenize_fn(batch):
    tok = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    # causal LM：labels = input_ids（collator 會處理 padding）
    tok["labels"] = tok["input_ids"].copy()
    return tok

ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ============================================================
# 5) TrainingArguments / Trainer
# ============================================================
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=True,     # CUDA 上建議開
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    data_collator=data_collator,
)

# ============================================================
# 6) Train + Save LoRA
# ============================================================
trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("✅ LoRA saved to:", OUT_DIR)

# =====================
# 7) 測試推論（直接用訓練完的 model）
# =====================
print("\n===== 測試推論 =====")
model.eval()

test_prompt = "### 指令:\n什麼是 ISO 27001 變更管理？\n\n### 回答:\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
