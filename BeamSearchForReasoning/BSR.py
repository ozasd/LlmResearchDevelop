"""
Script: BSR.py
Description: Listwise Verifier (Selection Mode) with Structured Output
Optimization: Ensure Output = Process (Steps) + Conclusion (Final)
"""

import os
import re
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# =========================
# 1. 全域設定
# =========================
TASK_NAME = "General Purpose - Selection Mode"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" 
LORA_DIR = None

USE_4BIT = False
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16

# ---- 核心參數 ----
K_CANDIDATES = 3
B_BEAM = 1              
MAX_DEPTH = 6           # 稍微增加深度，讓思考更完整

# ---- 生成參數 ----
STEP_MAX_NEW_TOKENS = 150
FINAL_MAX_NEW_TOKENS = 400
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.15

# ---- Verifier 設定 ----
VERIFIER_MODE = "llm"
VERIFIER_LLM_MAX_NEW_TOKENS = 300
VERIFIER_LLM_TEMPERATURE = 0.1

SEED = 42

QUESTION = "100 + 50 * 2 + 100 / 2"

# ---- 顯示設定 ----
SHOW_THINK = True
LOG_TRUNCATE = 600

# =========================
# 2. Prompt Templates
# =========================

GEN_SYSTEM_TEMPLATE = """
你是一個邏輯策劃者。你必須嚴格輸出 XML 標籤區塊，且只能輸出一個區塊。

【你只能輸出二選一（擇一）】
(1) <step>...單一步驟...</step>
(2) <final>...最終答案...</final>

【硬性規則】
- 只能輸出 1 個區塊：只能有一個 <step> 或一個 <final>
- 禁止在標籤外輸出任何字
- 內容要具體可執行；禁止重複 History
- 若已能直接得到最終答案，必須輸出 <final>
- 請用單行輸出（不要換行）

【範例】
<step>計算 50*2 得到 100。</step>
<final>結果 250</final>
"""

def build_gen_user(question, history_steps, step_id):
    # 這裡過濾掉 history 中的 <final> 標籤，避免生成時混淆，但保留 <step>
    clean_history = []
    for h in history_steps:
        if "<step>" in h:
            clean_history.append(h)
            
    history = "\n".join(clean_history).strip()
    history_block = f"已完成步驟：\n{history}\n" if history else "目前尚無步驟（第一步）。\n"
    return f"任務：{question}\n\n{history_block}\n請輸出下一步（One Step Only）："

VERIFIER_SYSTEM_PROMPT = """你是一個比較器（Verifier）。
你會看到 {k} 個候選輸出。

你的任務是：比較它們並選出唯一最佳的一個。

評估準則：
0. 若候選中存在「格式正確且能完整解答任務」的 <final>，優先選 <final>。
1. 優先：具體、邏輯連貫、貼合問題限制。
2. 淘汰：空泛、重複、偏題、格式錯誤。
3. 若是第一步：優先選能建立清楚方向與拆解架構者。
"""

VERIFIER_USER_TEMPLATE = """
[任務目標]
{question}

[目前進度]
{history}

[候選選項]
{candidates_block}

請輸出：
分析: 簡述優缺點
選擇: #N
"""

# =========================
# 3. 模型載入與工具
# =========================
def load_env_and_model():
    load_dotenv()
    # login(os.getenv("HF_TOKEN")) # 如有需要請取消註解
    torch.manual_seed(SEED)
    print(f"Loading model: {MODEL_ID}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
        quantization_config=quant_config,
    )
    model.eval()
    return tokenizer, model

def build_chat_prompt(tokenizer, system, user):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_seqs):
        super().__init__()
        self.stop_seqs = [seq for seq in stop_seqs if seq]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids is None or input_ids.numel() == 0:
            return False
        seq = input_ids[0].tolist()
        for stop in self.stop_seqs:
            n = len(stop)
            if n <= len(seq) and seq[-n:] == stop:
                return True
        return False

def build_stop_criteria(tokenizer):
    stops = ["</step>", "</final>"]
    stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]
    return StoppingCriteriaList([StopOnSubsequence(stop_ids)])

@torch.no_grad()
def generate_text(tokenizer, model, prompt, max_new_tokens, temp, rep_penalty, stopping_criteria=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_p=TOP_P,
        repetition_penalty=rep_penalty,
        do_sample=(temp > 0),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria,
    )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# =========================
# 4. Tag / parsing helpers
# =========================
TAG_BLOCK_RE = re.compile(r"<(step|final)\b[^>]*>.*?</\1>", re.I | re.S)

def normalize_one_line(s):
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tag_type(text):
    m = re.search(r"<(step|final)\b", text or "", re.I)
    return m.group(1).lower() if m else "step"

def extract_best_tag_block(text):
    s = (text or "").strip()
    if not s: return s
    
    # 1. 找完整區塊
    blocks = []
    for m in TAG_BLOCK_RE.finditer(s):
        blocks.append((m.group(1).lower(), m.group(0).strip()))
    
    if blocks:
        # 優先回傳 final
        for typ, blk in blocks:
            if typ == "final": return blk
        return blocks[0][1]

    # 2. 修補未閉合
    m_final = re.search(r"<final\b[^>]*>", s, re.I)
    m_step  = re.search(r"<step\b[^>]*>", s, re.I)
    m = m_final or m_step
    if not m: return s
    
    frag = s[m.start():].strip()
    typ = "final" if re.match(r"<final\b", frag, re.I) else "step"
    if f"</{typ}>" not in frag.lower():
        frag += f"</{typ}>"
    return frag

def ensure_single_tag_block(out):
    s = extract_best_tag_block(out).strip()
    if not re.search(r"<(step|final)\b", s, re.I):
        s = f"<step>{normalize_one_line(s)}</step>"
        return s
    s = re.sub(r"\s+", " ", s).strip()
    return extract_best_tag_block(s)

def strip_tag_content(tagged, tag):
    # 移除 tag 本身，保留內容
    content = re.sub(rf"</?{tag}.*?>", "", tagged or "", flags=re.I).strip()
    return content

def extract_choice_index(verifier_output, k):
    """
    從 verifier 輸出中抓取選擇的選項 (回傳 0-based index)。
    為了避免抓到「分析過程」中提到的數字，我們會由後往前 (reversed) 尋找最末段的結論。
    """
    text = (verifier_output or "").strip()
    
    # 策略 1: 優先尋找帶有明確指示詞的數字 (例如: 選擇: #2, choice 3, option 1)
    matches = re.findall(r"(?:選擇|selection|choice|option|選項)\s*[:：]?\s*#?\s*(\d+)", text, re.I)
    if matches:
        for num_str in reversed(matches):
            n = int(num_str)
            if 1 <= n <= k:
                return n - 1
                
    # 策略 2: 尋找帶有井字號的數字 (例如: #2)
    matches = re.findall(r"#\s*(\d+)", text)
    if matches:
        for num_str in reversed(matches):
            n = int(num_str)
            if 1 <= n <= k:
                return n - 1
                
    # 策略 3: 如果格式全跑掉，尋找最後出現的獨立數字 (1 到 k 之間)
    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        for num_str in reversed(matches):
            n = int(num_str)
            if 1 <= n <= k:
                return n - 1
                
    # 真的找不到才回傳 -1 (觸發 fallback)
    return -1

# =========================
# 5. Verifier & Logic
# =========================
def verify_batch_selection(tokenizer, model, question, history, candidates):
    cand_block = ""
    for i, cand in enumerate(candidates, 1):
        cand_block += f"# {i}: {normalize_one_line(cand)}\n"
    
    hist_text = "\n".join(history).strip() if history else "(第一輪)"
    
    sys_prompt = VERIFIER_SYSTEM_PROMPT.format(k=len(candidates))
    user_prompt = VERIFIER_USER_TEMPLATE.format(
        question=question, history=hist_text, candidates_block=cand_block
    )
    
    prompt = build_chat_prompt(tokenizer, sys_prompt, user_prompt)
    raw_output = generate_text(
        tokenizer, model, prompt, VERIFIER_LLM_MAX_NEW_TOKENS, 
        VERIFIER_LLM_TEMPERATURE, 1.05
    ).strip()
    
    winner_idx = extract_choice_index(raw_output, len(candidates))
    scores = [0.0] * len(candidates)
    if winner_idx >= 0:
        scores[winner_idx] = 10.0
    else:
        scores[0] = 5.0 # fallback default to first
        
    return scores, raw_output

def force_final_generation(tokenizer, model, question, history):
    history_text = "\n".join(history).strip()
    sys_prompt = "你是一個助手。請根據已完成的步驟，整合成最終完整回答。"
    user_prompt = f"""任務：{question}

已完成的步驟：
{history_text}

請停止新增步驟，直接輸出最終回答。
格式：<final>你的完整回答</final>
"""
    prompt = build_chat_prompt(tokenizer, sys_prompt, user_prompt)
    out = generate_text(
        tokenizer, model, prompt, FINAL_MAX_NEW_TOKENS, 0.5, 1.1, build_stop_criteria(tokenizer)
    ).strip()
    
    out = ensure_single_tag_block(out)
    if tag_type(out) != "final":
        content = strip_tag_content(out, "step")
        out = f"<final>{content}</final>"
    return out

# =========================
# 6. 主流程 (Logic Logic Logic)
# =========================
def run_beam_reasoning(tokenizer, model, question):
    current_steps = []
    final_result_text = None
    
    print(f"Start reasoning. max_depth={MAX_DEPTH}", flush=True)
    stop_criteria = build_stop_criteria(tokenizer)

    for t in range(1, MAX_DEPTH + 1):
        # Generate
        raw_candidates_txt = []
        for _ in range(K_CANDIDATES):
            prompt = build_chat_prompt(tokenizer, GEN_SYSTEM_TEMPLATE, build_gen_user(question, current_steps, t))
            out = generate_text(tokenizer, model, prompt, STEP_MAX_NEW_TOKENS, TEMPERATURE, REPETITION_PENALTY, stop_criteria)
            raw_candidates_txt.append(ensure_single_tag_block(out))

        # Verify
        scores, verifier_comment = verify_batch_selection(tokenizer, model, question, current_steps, raw_candidates_txt)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        winner_txt = raw_candidates_txt[best_idx]

        # Log
        if SHOW_THINK:
            print(f"\n=== Depth {t}/{MAX_DEPTH} ===", flush=True)
            for i, txt in enumerate(raw_candidates_txt, 1):
                mark = "WIN" if (i - 1) == best_idx else "REJ"
                print(f"{mark} #{i}: {normalize_one_line(txt)[:100]}...", flush=True)
            
            # ===== 印出 Verifier 的完整分析 =====
            print("-" * 60, flush=True)
            print("[Verifier Analysis]", flush=True)
            print((verifier_comment or "").strip(), flush=True)
            print("-" * 60, flush=True)
            print(f">>> Final Choice: #{best_idx+1}", flush=True)
            
        # Logic Branching
        winner_type = tag_type(winner_txt)
        
        if winner_type == "final":
            # 這是我們想要邏輯：這不是 Step，這是 Final
            final_result_text = winner_txt
            # 我們把它存在結果變數，不放入 steps 列表（或者放進去但標記它是 final）
            # 為了主程式好處理，這裡我們直接 return 
            return {"steps": current_steps, "final_text": final_result_text, "status": "completed"}
        else:
            # 這是 Step，加入歷史
            current_steps.append(winner_txt)

    # Max depth reached
    print("\n[WARN] Max depth reached; forcing final.", flush=True)
    forced_final = force_final_generation(tokenizer, model, question, current_steps)
    return {"steps": current_steps, "final_text": forced_final, "status": "forced_final"}

# =========================
# 7. Main (Structured Output)
# =========================
def main():
    tokenizer, model = load_env_and_model()

    print("\n" + "=" * 60)
    print(f"Task: {TASK_NAME}")
    print(f"Question: {QUESTION}")
    print("=" * 60)

    result = run_beam_reasoning(tokenizer, model, QUESTION)

    # ---------------------------------------------------------
    # 核心修改：嚴格分離 Process (Think) 與 Result (Results)
    # ---------------------------------------------------------
    
    steps_list = result["steps"]
    final_text = result["final_text"]

    print("\n" + "=" * 60)
    print("=== FINAL STRUCTURED OUTPUT ===", flush=True)
    
    # 1. 輸出 <think>：只包含過程步驟
    print("<think>", flush=True)
    if not steps_list:
        print("(No intermediate steps generated, jumped straight to final)", flush=True)
    else:
        for i, step_raw in enumerate(steps_list, 1):
            # 清洗 tag，只留內容
            content = strip_tag_content(step_raw, "step")
            content = normalize_one_line(content)
            print(f"{i}. {content}", flush=True)
    print("</think>", flush=True)

    # 2. 輸出 <results>：只包含最終結果
    # 這裡的 final_text 應該是包含了步驟邏輯後的產出
    print("\n<results>", flush=True)
    
    if final_result_text := strip_tag_content(final_text, "final"):
        print(normalize_one_line(final_result_text), flush=True)
    else:
        print("（模型未能生成有效結果）", flush=True)
        
    print("</results>", flush=True)
    print("=" * 60)

if __name__ == "__main__":
    main()