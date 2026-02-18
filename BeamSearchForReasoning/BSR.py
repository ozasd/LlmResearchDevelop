"""

Script: reason_beam_select.py

Description: Listwise Verifier (Selection Mode) with Structured Output



通用型推理引擎（Selection Mode / Listwise Verifier）。



重點強化：

1) 3B 輸出 <step>/<final> 不穩 → 以「硬範本」+「停止條件」+「解析/修補」強制格式

2) 只要選到 <final> → 立刻 early stop（不跑滿 MAX_DEPTH）

3) 解析策略：優先擷取完整 <final>...</final>，避免「先 step 後 final」被截到 step

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



USE_4BIT = True

DEVICE_MAP = "auto"

TORCH_DTYPE = torch.float16



# ---- 核心參數 ----

K_CANDIDATES = 3

B_BEAM = 1              # 保留欄位，但此模式下不使用多 beam

MAX_DEPTH = 8



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



QUESTION = "100 + 50 * 2 + 50"



# ---- 顯示設定 ----

SHOW_THINK = True

LOG_TRUNCATE = 240  # Console 顯示每個候選最多顯示字元數





# =========================

# 2. Prompt Templates

# =========================



# 生成端：硬性格式（3B 會穩很多）

GEN_SYSTEM_TEMPLATE = """

你是一個邏輯策劃者。你必須嚴格輸出 XML 標籤區塊，且只能輸出一個區塊。



【你只能輸出二選一（擇一）】

(1) <step>...單一步驟...</step>

(2) <final>...最終答案...</final>





【硬性規則（違反視為錯誤）】



- 只能輸出 1 個區塊：只能有一個 <step> 或一個 <final>

- 禁止在標籤外輸出任何字（包含解釋、空話、前後綴）

- 內容要具體可執行；禁止重複 History

- 若已能直接得到最終答案，必須輸出 <final>，不要再輸出 <step>

- 請用單行輸出（不要換行）



【正確示例】

<step>先計算乘法項 50*2 得到 100。</step>

<final>結果250</final>



【錯誤示例（禁止）】

好的，我會這樣做：<step>...</step> <step>...</step><step>...</step> <step>...</step> <final>...</final>（在 tag 外多字）



"""





def build_gen_user(question: str, history_steps: list[str], step_id: int) -> str:

    history = "\n".join(history_steps).strip()

    history_block = f"已完成：\n{history}\n" if history else "目前尚無步驟（第一步）。\n"

    return f"任務：{question}\n\n{history_block}\n請輸出下一步（One Step Only）："





# Verifier：明確指示「若有高品質 <final>，優先選 <final>」

VERIFIER_SYSTEM_PROMPT = """你是一個比較器（Verifier）。

你會看到 {k} 個候選輸出（每個候選必須是唯一一個區塊：<step>...</step> 或 <final>...</final>）。



你的任務是：比較它們並選出唯一最佳的一個。



評估準則：

0. 若候選中存在「格式正確且能完整解答任務」的 <final>，優先選 <final>（避免不必要的後續步驟）。

1. 優先：具體、可執行、邏輯連貫、貼合問題限制。

2. 淘汰：空泛、重複、偏題、格式錯誤、無法推進。

3. 若是第一步：優先選能建立清楚方向與拆解架構者。

若全部品質偏低，仍需選出相對最佳者，並指出主要問題。

"""



VERIFIER_USER_TEMPLATE = """

[任務目標]

{question}



[目前進度（History）]

{history}



[候選選項]

{candidates_block}



請輸出：

分析: 逐一指出 #1, #2, #3... 的優缺點

選擇: #N

"""





# =========================

# 3. 模型載入與工具

# =========================

def load_env_and_model():

    load_dotenv()

    if os.getenv("HF_TOKEN"):

        login(os.getenv("HF_TOKEN"))



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



    if LORA_DIR:

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, LORA_DIR)

        print(f"LoRA loaded: {LORA_DIR}", flush=True)



    return tokenizer, model





def build_chat_prompt(tokenizer, system: str, user: str) -> str:

    messages = [{"role": "system", "content": system},

                {"role": "user", "content": user}]

    return tokenizer.apply_chat_template(

        messages, tokenize=False, add_generation_prompt=True

    )





# =========================

# 3.1 停止條件：看到 </step> 或 </final> 就停（避免 tag 外亂跑）

# =========================

class StopOnSubsequence(StoppingCriteria):

    def __init__(self, stop_seqs: list[list[int]]):

        super().__init__()

        self.stop_seqs = [seq for seq in stop_seqs if seq]



    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        # input_ids: [batch, seq_len]

        if input_ids is None or input_ids.numel() == 0:

            return False

        seq = input_ids[0].tolist()

        for stop in self.stop_seqs:

            n = len(stop)

            if n <= len(seq) and seq[-n:] == stop:

                return True

        return False





def build_stop_criteria(tokenizer) -> StoppingCriteriaList:

    # 不加 special tokens，純粹匹配字面 tag

    stops = ["</step>", "</final>"]

    stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]

    return StoppingCriteriaList([StopOnSubsequence(stop_ids)])





@torch.no_grad()

def generate_text(

    tokenizer,

    model,

    prompt: str,

    max_new_tokens: int,

    temp: float,

    rep_penalty: float,

    stopping_criteria: StoppingCriteriaList | None = None,

) -> str:

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



def normalize_one_line(s: str) -> str:

    s = (s or "").strip()

    s = re.sub(r"\s+", " ", s)

    return s



def tag_type(text: str) -> str:

    # 只要包含 <final 就視為 final（因為我們會優先抽 final）

    m = re.search(r"<(step|final)\b", text or "", re.I)

    return m.group(1).lower() if m else "step"



def extract_best_tag_block(text: str) -> str:

    """

    優先抽出第一個完整 <final>...</final>，

    若沒有，再抽第一個完整 <step>...</step>。

    若都沒有完整 close tag，則從第一個 open tag 起截到結尾並補 close tag。

    """

    s = (text or "").strip()

    if not s:

        return s



    # 1) 找所有完整區塊

    blocks: list[tuple[str, str]] = []

    for m in TAG_BLOCK_RE.finditer(s):

        typ = m.group(1).lower()

        blk = m.group(0).strip()

        blocks.append((typ, blk))



    if blocks:

        # 先挑 final

        for typ, blk in blocks:

            if typ == "final":

                return blk

        # 否則回第一個 step

        return blocks[0][1]



    # 2) 沒有完整 close tag：找第一個 open tag（優先 final）

    m_final = re.search(r"<final\b[^>]*>", s, re.I)

    m_step  = re.search(r"<step\b[^>]*>", s, re.I)



    m = m_final or m_step

    if not m:

        return s



    frag = s[m.start():].strip()

    typ = "final" if re.match(r"<final\b", frag, re.I) else "step"

    close = f"</{typ}>"

    if close.lower() not in frag.lower():

        frag = frag + close

    return frag



def ensure_single_tag_block(out: str) -> str:

    """

    確保輸出最後只剩「一個」<step> 或 <final> 區塊。

    1) 先抽 best block（優先 final）

    2) 移除區塊內換行 → 變單行

    3) 若還是沒有 tag，就包成 <step>

    """

    s = extract_best_tag_block(out)

    s = s.strip()



    if not re.search(r"<(step|final)\b", s, re.I):

        s = f"<step>{normalize_one_line(s)}</step>"

        return s



    # 強制單行（但保留 tag）

    # 把 tag 之間的內容壓成單行

    s = re.sub(r"\s+", " ", s).strip()



    # 避免同一行中出現多個區塊：再抽一次（保險）

    s = extract_best_tag_block(s)

    s = re.sub(r"\s+", " ", s).strip()



    return s



def strip_tag_content(tagged: str, tag: str) -> str:

    return re.sub(rf"</?{tag}.*?>", "", tagged or "", flags=re.I).strip()





def extract_choice_index(verifier_output: str, k: int) -> int:

    """

    從 verifier 輸出抓 #N。

    支援：

      - 選擇: #3

      - Choice: 3

      - Selection: Option 2

    回傳 0-based index；抓不到回 -1。

    """

    text = (verifier_output or "").strip()



    m = re.search(r"(選擇|selection|choice)\s*[:：]?\s*#?\s*(\d+)", text, re.I)

    if m:

        n = int(m.group(2))

        if 1 <= n <= k:

            return n - 1



    m = re.search(r"(option|選項)\s*#?\s*(\d+)", text, re.I)

    if m:

        n = int(m.group(2))

        if 1 <= n <= k:

            return n - 1



    nums = re.findall(r"#\s*(\d+)", text)

    if nums:

        n = int(nums[-1])

        if 1 <= n <= k:

            return n - 1



    return -1





# =========================

# 5. Verifier 與強制結算

# =========================

def verify_batch_selection(tokenizer, model, question: str, history: list[str], candidates: list[str]):

    """

    一次將所有候選丟給 LLM，讓它選出 Winner。

    回傳：scores(list[float]), verifier_text(str)

    """

    cand_block = ""

    for i, cand in enumerate(candidates, 1):

        clean_cand = normalize_one_line(cand)

        cand_block += f"# {i}: {clean_cand}\n"



    hist_text = "\n".join(history).strip() if history else "(無 - 第一輪)"



    sys_prompt = VERIFIER_SYSTEM_PROMPT.format(k=len(candidates))

    user_prompt = VERIFIER_USER_TEMPLATE.format(

        question=question,

        history=hist_text,

        candidates_block=cand_block

    )



    prompt = build_chat_prompt(tokenizer, sys_prompt, user_prompt)



    raw_output = generate_text(

        tokenizer, model, prompt,

        max_new_tokens=VERIFIER_LLM_MAX_NEW_TOKENS,

        temp=VERIFIER_LLM_TEMPERATURE,

        rep_penalty=1.05,

        stopping_criteria=None,  # verifier 不需要 tag stop

    ).strip()



    winner_idx = extract_choice_index(raw_output, len(candidates))



    scores = [0.0] * len(candidates)

    if winner_idx >= 0:

        scores[winner_idx] = 10.0

    else:

        if scores:

            scores[0] = 5.0  # fallback



    return scores, raw_output





def force_final_generation(tokenizer, model, question: str, history: list[str]) -> str:

    """

    當步數耗盡時，強制模型根據現有步驟做總結。

    """

    history_text = "\n".join(history).strip()



    sys_prompt = "你是一個助手。請根據已完成的步驟，整合成最終完整回答。"

    user_prompt = f"""任務：{question}



已完成的步驟：

{history_text}



請停止新增步驟，直接輸出最終回答。

格式：<final>你的完整回答</final>

禁止在標籤外輸出任何字，請用單行輸出。

"""



    prompt = build_chat_prompt(tokenizer, sys_prompt, user_prompt)



    out = generate_text(

        tokenizer, model, prompt,

        max_new_tokens=FINAL_MAX_NEW_TOKENS,

        temp=0.5,

        rep_penalty=1.1,

        stopping_criteria=build_stop_criteria(tokenizer),

    ).strip()



    out = ensure_single_tag_block(out)

    if tag_type(out) != "final":

        # 仍然不是 final：硬包一次

        content = normalize_one_line(strip_tag_content(out, "step") or out)

        out = f"<final>{content}</final>"

    return extract_best_tag_block(out)





# =========================

# 6. 主流程

# =========================

def run_beam_reasoning(tokenizer, model, question: str):

    current_steps: list[str] = []



    print(f"Start reasoning. max_depth={MAX_DEPTH}, k={K_CANDIDATES}", flush=True)



    stop_criteria = build_stop_criteria(tokenizer)



    for t in range(1, MAX_DEPTH + 1):

        # 1) Generate candidates

        raw_candidates_txt: list[str] = []

        for _ in range(K_CANDIDATES):

            gen_sys = GEN_SYSTEM_TEMPLATE

            gen_user = build_gen_user(question, current_steps, t)

            prompt = build_chat_prompt(tokenizer, gen_sys, gen_user)



            out = generate_text(

                tokenizer, model, prompt,

                max_new_tokens=STEP_MAX_NEW_TOKENS,

                temp=TEMPERATURE,

                rep_penalty=REPETITION_PENALTY,

                stopping_criteria=stop_criteria,

            )



            # 強制修補成唯一 tag block（優先 final）

            out = ensure_single_tag_block(out)

            raw_candidates_txt.append(out)



        # 2) Verify (listwise selection)

        scores, verifier_comment = verify_batch_selection(

            tokenizer, model, question, current_steps, raw_candidates_txt

        )



        # 3) Select winner

        best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0

        winner_txt = raw_candidates_txt[best_idx]



        # 4) Logging (plain text, no emoji)

        if SHOW_THINK:

            print("\n" + "=" * 80, flush=True)

            print(f"[Depth {t}/{MAX_DEPTH}] selection (compare {K_CANDIDATES} candidates)", flush=True)

            print("=" * 80, flush=True)



            for i, txt in enumerate(raw_candidates_txt, 1):

                one_line = normalize_one_line(txt)

                if len(one_line) > LOG_TRUNCATE:

                    one_line = one_line[:LOG_TRUNCATE] + "..."

                mark = "WIN" if (i - 1) == best_idx else "REJ"

                print(f"{mark} #{i}: {one_line}", flush=True)



            print("-" * 60, flush=True)

            print("[Verifier output]", flush=True)

            print((verifier_comment or "").strip(), flush=True)

            print("=" * 80, flush=True)



        # 5) Append to history

        current_steps.append(winner_txt)



        # 6) Early stop if final

        if tag_type(winner_txt) == "final":

            return {"steps": current_steps, "final_text": winner_txt, "status": "completed"}



    # Max depth reached, force final

    print("\n[WARN] max depth reached; forcing final.", flush=True)

    forced_final = force_final_generation(tokenizer, model, question, current_steps)

    return {"steps": current_steps, "final_text": forced_final, "status": "forced_final"}





# =========================

# Main (Output Formatting)

# =========================

def main():

    tokenizer, model = load_env_and_model()



    print("\n" + "=" * 60, flush=True)

    print(f"Task: {TASK_NAME}", flush=True)

    print(f"Question: {QUESTION}", flush=True)

    print("=" * 60, flush=True)



    result = run_beam_reasoning(tokenizer, model, QUESTION)



    think_content: list[str] = []

    results_content = result.get("final_text", "")



    # 解析 steps -> think

    for step in result["steps"]:

        s = (step or "").strip()



        # final 不放進 think

        if re.search(r"<final\b", s, re.I):

            if not results_content:

                results_content = s

            continue



        # 去掉 step tag

        clean_step = strip_tag_content(s, "step")

        clean_step = normalize_one_line(clean_step)

        if clean_step:

            think_content.append(clean_step)



    # results 去 tag

    clean_results = strip_tag_content(results_content, "final")

    clean_results = normalize_one_line(clean_results)

    if not clean_results:

        clean_results = "（模型未能生成最終結果）"



    print("\n" + "=" * 60, flush=True)

    print("=== FINAL STRUCTURED OUTPUT ===", flush=True)



    print("<think>", flush=True)

    for i, thought in enumerate(think_content, 1):

        print(f"{i}. {thought}", flush=True)

    print("</think>", flush=True)



    print("\n<results>", flush=True)

    print(clean_results, flush=True)

    print("</results>", flush=True)

    print("=" * 60, flush=True)





if __name__ == "__main__":

    main()