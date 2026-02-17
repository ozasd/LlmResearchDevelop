import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from dotenv import load_dotenv

# 引入剛剛寫好的資料準備模組
from prepare_data import get_general_dataset

# ============================================================
# 基本設定
# ============================================================
MODEL_ID   = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH  = "./customerService.json" ## 甜點客服資料集
OUT_DIR    = "./llama3_cs_lora"       ## 輸出路徑改為 cs_lora

MAX_LEN    = 384
BATCH_SIZE = 1
GRAD_ACC   = 8
EPOCHS     = 3    # 客服對話通常比較簡單，3 Epochs 通常足夠
LR         = 1e-4
SEED       = 42
MIX_RATIO  = 0.2  # 混合 20% 通用資料維持對話流暢度

# ============================================================
# HF 登入
# ============================================================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

torch.manual_seed(SEED)

# ============================================================
# 1) 載入與混合資料 (Modified)
# ============================================================
print(">>> 步驟 1: 準備資料...")

# A. 載入甜點客服資料 (變數名稱由 ds_iso 改為 ds_cs)
ds_cs = load_dataset("json", data_files=DATA_PATH, split="train")

def format_cs(x):
    inst = (x.get("instruction") or "").strip()
    out  = (x.get("output") or "").strip()
    return f"### 指令:\n{inst}\n\n### 回答:\n{out}"

# 格式化並只保留 text 欄位
ds_cs = ds_cs.map(lambda x: {"text": format_cs(x)})
ds_cs = ds_cs.remove_columns([c for c in ds_cs.column_names if c != "text"])
print(f"   甜點客服資料載入完成: {len(ds_cs)} 筆")

# B. 載入通用資料 (透過 prepare_data.py)
# 客服機器人非常需要通用資料，不然會變得只會回答產品，不會寒暄
num_general = int(len(ds_cs) * MIX_RATIO)
num_general = max(1, num_general) 

ds_gen = get_general_dataset(num_samples=num_general, seed=SEED)

# C. 合併資料集
if ds_gen:
    ds = concatenate_datasets([ds_cs, ds_gen])
    print(f"   資料合併完成。總筆數: {len(ds)} (客服: {len(ds_cs)} + 通用: {len(ds_gen)})")
else:
    ds = ds_cs
    print("   警告: 通用資料載入失敗，僅使用客服資料。")

# D. 再次打亂
ds = ds.shuffle(seed=SEED)

# ============================================================
# 2) Tokenizer / Model (QLoRA 4-bit)
# ============================================================
print(">>> 步驟 2: 載入模型與 Tokenizer...")
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
# 4) 手動 tokenize + truncate
# ============================================================
print(">>> 步驟 3: Tokenization...")
def tokenize_fn(batch):
    tok = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
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
print(">>> 步驟 4: 開始訓練...")
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
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
print(f"✅ LoRA 模型已儲存至: {OUT_DIR}")

# =====================
# 7) 測試推論
# =====================
print("\n===== 測試推論 (甜點情境) =====")
model.eval()

# 修改測試問題為甜點相關
test_prompt = "### 指令:\n請問草莓蛋糕可以宅配嗎？如果運送壞掉怎麼辦？\n\n### 回答:\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        
        # === 修改這裡 ===
        temperature=0.6,      # 稍微調低一點，讓回答更穩定 (原本 0.7)
        top_p=0.9,
        do_sample=True,
        
        # ★ 關鍵參數：重複懲罰
        repetition_penalty=1.2,  # 設定 1.1 ~ 1.2 都可以，強迫它不講重複的話
        
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# 這裡建議加上 .split("### 回答:")[-1]，只印出回答部分，版面比較乾淨
response = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Q: {config['test_prompt']}")
# print(f"A: {response}") # 原本的印法

# 優化後的印法 (只印出生成的後半段)
if "### 回答:" in response:
    print(f"A: {response.split('### 回答:')[-1].strip()}")
else:
    print(f"A: {response}")