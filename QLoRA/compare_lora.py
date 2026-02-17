import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel

# =========================
# 你要改的地方
# =========================
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"   # 你原生模型
LORA_DIR      = r"./llama3_cs_lora"                   # ★ 改成您的甜點客服 output 目錄 (或是 iso)
QUESTION      = "請問草莓蛋糕可以宅配嗎？如果壞掉怎麼辦？"   # ★ 測試問題

# 公平比較：固定隨機種子
SEED = 42

# =========================
# ★ 優化後的解碼參數 (解決跳針問題)
# =========================
GEN_KWARGS = dict(
    max_new_tokens=220,
    # do_sample=False,    # 原本是 False (Greedy)，容易跳針
    do_sample=True,       # 改成 True，讓回答自然一點
    temperature=0.6,      # 低溫，保持穩定
    top_p=0.9,
    repetition_penalty=1.2, # ★ 關鍵：強制懲罰重複的句子
)

# =========================
# 共用：建立 prompt
# =========================
def build_prompt(tokenizer, user_question):
    # ★ 關鍵修改：
    # 由於您訓練時是用 "### 指令: ... ### 回答:"
    # 推論時最好用 "一模一樣" 的格式，LoRA 效果才會最好！
    # 如果用 chat_template，模型可能會切換回原本的說話方式，看不出微調效果。
    
    pattern = f"### 指令:\n{user_question}\n\n### 回答:\n"
    return pattern

# =========================
# 共用：生成
# =========================
@torch.no_grad()
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **GEN_KWARGS
    )
    # 只取生成的回答部分 (去掉前面的 Prompt)
    full_output = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # 幫您做字串切割，只印出回答
    if "### 回答:" in full_output:
        return full_output.split("### 回答:")[-1].strip()
    return full_output

def cleanup(*objs):
    for o in objs:
        try:
            del o
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    set_seed(SEED)

    # 4-bit 量化設定
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = build_prompt(tokenizer, QUESTION)
    print(f"使用的 Prompt 格式:\n{prompt}")

    # =========================
    # 1) 原生 Base 回答
    # =========================
    print("正在載入 Base Model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        device_map="auto"
    )
    base.eval()

    print("生成 Base 回答中...")
    ans_base = generate(base, tokenizer, prompt)

    # =========================
    # 2) 套 LoRA 後回答
    # =========================
    print("正在掛載 LoRA Adapter...")
    # 這裡直接在原本的 base 上面掛載，省記憶體
    lora = PeftModel.from_pretrained(base, LORA_DIR)
    lora.eval()

    print("生成 LoRA 回答中...")
    ans_lora = generate(lora, tokenizer, prompt)

    # =========================
    # 印出比較結果
    # =========================
    print("\n" + "="*80)
    print("【問題】", QUESTION)
    print("="*80)
    
    print("\n----- (A) 原生 Base (沒訓練過) -----\n")
    print(ans_base)
    
    print("\n" + "-"*40 + "\n")
    
    print("----- (B) Base + LoRA (微調後) -----\n")
    print(ans_lora)
    
    print("\n" + "="*80 + "\n")

    cleanup(lora, base, tokenizer)

if __name__ == "__main__":
    main()