"""
Script: train_lora.py
Description: 通用型 Llama 3.2 LoRA 微調腳本 (包含災難性遺忘「雙重測試」)
Usage: 直接修改 main() 裡的設定變數，然後執行 python train_lora.py
"""

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

# ============================================================
# 嘗試引入自定義的抗遺忘資料模組
# ============================================================
try:
    from prepare_data import get_general_dataset
except ImportError:
    print("警告: 找不到 prepare_data.py，將跳過通用資料混合。")
    get_general_dataset = None

# ============================================================
# 核心訓練邏輯
# ============================================================
def run_training(config):
    print(f"\n[任務啟動] {config['task_name']}")

    # 1. 環境設定
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)
    torch.manual_seed(config['seed'])

    # 2. 準備資料
    print("\n>>> [Step 1] 載入資料...")
    try:
        ds_domain = load_dataset("json", data_files=config['data_path'], split="train")
    except Exception as e:
        print(f"資料載入失敗: {e}")
        return

    def format_instruction(x):
        inst = (x.get("instruction") or "").strip()
        out = (x.get("output") or "").strip()
        return f"### 指令:\n{inst}\n\n### 回答:\n{out}"

    ds_domain = ds_domain.map(lambda x: {"text": format_instruction(x)})
    ds_domain = ds_domain.remove_columns([c for c in ds_domain.column_names if c != "text"])

    # 混合通用資料 (Anti-Forgetting)
    ds_final = ds_domain
    if get_general_dataset and config.get('mix_ratio', 0) > 0:
        num_gen = int(len(ds_domain) * config['mix_ratio'])
        num_gen = max(1, num_gen)
        print(f"混合通用資料防遺忘 (比例 {config['mix_ratio']}, 約 {num_gen} 筆)...")
        ds_gen = get_general_dataset(num_samples=num_gen, seed=config['seed'])
        if ds_gen:
            ds_final = concatenate_datasets([ds_domain, ds_gen])

    ds_final = ds_final.shuffle(seed=config['seed'])
    print(f"最終訓練集: {len(ds_final)} 筆")

    # 3. 載入模型
    print("\n>>> [Step 2] 載入模型 (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], quantization_config=bnb_config, device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA 設定
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    # 5. Tokenization
    def tokenize_fn(batch):
        tok = tokenizer(
            batch["text"], truncation=True, max_length=config['max_len'], padding=False
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds_train = ds_final.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. 開始訓練
    print(f"\n>>> [Step 3] 開始訓練 (Epochs: {config['epochs']})...")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=config['output_dir'],
            per_device_train_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['grad_acc'],
            num_train_epochs=config['epochs'],
            learning_rate=config['lr'],
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            fp16=True,
            bf16=False,
        ),
        train_dataset=ds_train,
        data_collator=data_collator,
    )
    trainer.train()

    # 7. 存檔
    model.save_pretrained(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    print(f"模型已儲存至: {config['output_dir']}")

    # ============================================================
    # 雙重測試 (Double Check)
    # ============================================================
    print("\n" + "=" * 60)
    print("雙重驗證：領域知識 vs 災難性遺忘測試")
    print("=" * 60)

    model.eval()

    # 定義測試清單：(測試名稱, 測試問題)
    test_cases = [
        ("領域測試 (看有沒有學會)", config['test_prompt_domain']),
        ("通用測試 (看有沒有變笨)", config['test_prompt_general'])
    ]

    for test_name, question in test_cases:
        inputs = tokenizer(
            f"### 指令:\n{question}\n\n### 回答:\n", return_tensors="pt"
        ).to(model.device)

        # 取得設定中的防跳針參數
        rep_penalty = config.get("repetition_penalty", 1.2)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=rep_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0], skip_special_tokens=True)

        # 格式化輸出
        print(f"\n--- {test_name} ---")
        print(f"Q: {question}")
        if "### 回答:" in response:
            clean_ans = response.split("### 回答:")[-1].strip()
            print(f"A: {clean_ans}")
        else:
            print(f"A: {response}")

    print("\n" + "=" * 60)


# ============================================================
# Main: 請在這裡調整參數與切換任務
# ============================================================
def main():

    # 【配置 A】 ISO 27001 資安顧問
    # config = {
    #     "task_name": "ISO 27001 Fine-tuning",
    #     "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    #     "data_path": "./ISO27001.json",
    #     "output_dir": "./llama3_iso_lora",
    #     "epochs": 5,
    #     "batch_size": 1,
    #     "grad_acc": 8,
    #     "lr": 1e-4,
    #     "max_len": 512,
    #     "mix_ratio": 0.2,  # 混合通用資料
    #     "seed": 42,
    #     "repetition_penalty": 1.15,

    #     # 雙重測試問題
    #     "test_prompt_domain": "什麼是 ISO 27001 變更管理？",  # 測專業
    #     "test_prompt_general": "請列出太陽系有哪些行星？並簡單介紹地球。"  # 測遺忘
    # }

    # 【配置 B】 甜點客服機器人
    config = {
        "task_name": "Dessert Customer Service",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "data_path": "./customerService.json",
        "output_dir": "./llama3_cs_lora",
        "epochs": 3,
        "batch_size": 1,
        "grad_acc": 8,
        "lr": 1e-4,
        "max_len": 384,
        "mix_ratio": 0.2,  # 混合通用資料
        "seed": 42,
        "repetition_penalty": 1.2,

        # 雙重測試問題
        "test_prompt_domain": "請問草莓蛋糕可以宅配嗎？如果壞掉怎麼辦？",  # 測客服
        "test_prompt_general": "這禮拜天氣如何？適合出去玩嗎？"  # 測閒聊
    }

    run_training(config)


if __name__ == "__main__":
    main()
