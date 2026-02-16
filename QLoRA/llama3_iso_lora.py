import os, gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel

# =========================
# 你要改的地方
# =========================
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.2-3B-Instruct"   # 你原生模型
LORA_DIR      = r"./llama3_iso_lora"                  # 你的 LoRA 資料夾
QUESTION      = "什麼是 ISO 27001 變更管理？"          # 同一題拿來比較

# （建議）公平比較：固定隨機種子 + deterministic 生成
SEED = 42

# 解碼參數（兩邊要一樣）
GEN_KWARGS = dict(
    max_new_tokens=220,
    do_sample=False,     # 關掉抽樣，讓比較更穩定
    temperature=0.0,
    top_p=1.0,
)

# =========================
# 共用：建立 prompt（優先用 chat template）
# =========================
def build_prompt(tokenizer, user_question: str) -> str:
    messages = [
        {"role": "system", "content": "你是資深 ISO 27001:2022 資安稽核員與 IT 技術顧問，請用繁體中文、工程師情境回答，必要時引用 Annex A 控制項精神。"},
        {"role": "user", "content": user_question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # fallback：你之前訓練用的格式
    return f"### 指令:\n{user_question}\n\n### 回答:\n"

# =========================
# 共用：生成
# =========================
@torch.no_grad()
def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **GEN_KWARGS
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

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

    # 4-bit 量化設定（和你訓練一致）
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # tokenizer 建議從 LoRA 資料夾載（你那邊有 chat_template.jinja）
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = build_prompt(tokenizer, QUESTION)

    # =========================
    # 1) 原生 base 回答
    # =========================
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        # device_map={"": 0} if torch.cuda.is_available() else "auto",
        device_map="cpu"
    )
    base.eval()

    ans_base = generate(base, tokenizer, prompt)

    # =========================
    # 2) 套 LoRA 後回答（同一個 base 直接套 adapter）
    # =========================
    lora = PeftModel.from_pretrained(base, LORA_DIR)
    lora.eval()

    ans_lora = generate(lora, tokenizer, prompt)

    # =========================
    # 印出比較結果
    # =========================
    print("\n" + "="*80)
    print("【問題】", QUESTION)
    print("="*80)
    print("\n----- (A) 原生 Base -----\n")
    print(ans_base)
    print("\n----- (B) Base + LoRA (Fine-tuned) -----\n")
    print(ans_lora)
    print("\n" + "="*80 + "\n")

    cleanup(lora, base, tokenizer)

if __name__ == "__main__":
    main()
