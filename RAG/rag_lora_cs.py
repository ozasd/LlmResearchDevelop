"""
Cutting Knowledge Date: December 2023
Today Date: 20 Feb 2026

你是甜點店客服助理。請你依據目前客服資料進行回答，禁止自行編造。客服資料：
門市資訊｜電話｜常見問法: 電話,聯絡電話,門市電話,打電話,怎麼聯絡｜答案: 0908137159
門市資訊｜地址｜常見問法: 店址,門市地址,地址在哪,店在哪裡,怎麼去,地點在哪｜答案: 台北市大安區和平東路三段123號
聯絡方式｜線上客服時間｜常見問法: 線上客服,客服時間,真人客服,客服幾點到幾點,有人回嗎｜答案: 每日 09:00-18:00
聯絡方式｜客服信箱｜常見問法: 客服信箱,email,電子郵件,寫信給你們,怎麼寄信｜答案: ozasd6565@gmail.com
門市資訊｜營業時間｜常見問法: 幾點開,幾點關,營業到幾點,今天有開嗎,營業時間｜答案: 週一至週日 10:00-21:00
付款方式｜可用支付｜常見問法: 怎麼付款,支付方式,可刷卡嗎,可以用LINE Pay嗎,可以轉帳嗎,匯款｜答案: 本店接受 信用卡、LINE Pay、銀行轉帳 付款。user  

問題：請問一下你們店家電話與店家地址?assistant

店家電話：0908137159
店家地址：台北市大安區和平東路三段123號

"""
import os
import gc
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# =========================
# 基本設定
# =========================
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LORA_DIR      = r"./llama3_cs_lora"
QUESTION      = "請問一下你們店家電話與店家地址?"

SEED = 42
TOP_K = 6
COLLECTION_NAME = "universal_docs"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

GEN_KWARGS = dict(
    max_new_tokens=80,
    do_sample=False,
    repetition_penalty=1.1,
)

def build_chat_prompt(tokenizer, question, context):
    messages = [
        {
            "role": "system",
            "content": (
                "你是甜點店客服助理。請你依據目前客服資料進行回答，禁止自行編造。"
                f"客服資料：\n{context}\n\n"

            )
        },
        {
            "role": "user",
            "content": (
                f"問題：{question}\n\n"
            )
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

@torch.no_grad()
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **GEN_KWARGS
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.strip()

def cleanup(*objs):
    for o in objs:
        try:
            del o
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def gpu_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)

def main():
    set_seed(SEED)

    # =========================
    # 1) 向量檢索 (只查 CS)
    # =========================
    # print("Querying vector DB...")

    embed_model = SentenceTransformer("BAAI/bge-large-zh", device="cpu")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    query_embedding = embed_model.encode([QUESTION]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        where={"source": "CS"}
    )

    documents = results["documents"][0]
    context = "\n".join(documents)

    # print("Retrieved context:\n", context)
    cleanup(embed_model)

    # =========================
    # 2) 載入模型 (4bit)
    # =========================
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # print("Loading base model...")

    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        base.eval()
    except Exception:
        max_mem = {"cpu": "48GiB"}
        if torch.cuda.is_available():
            vram = gpu_vram_gb()
            gpu_allow = max(2.0, vram * 0.75)
            max_mem[0] = f"{gpu_allow:.1f}GiB"

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            max_memory=max_mem,
            llm_int8_enable_fp32_cpu_offload=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        base.eval()

    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()

    # =========================
    # 3) 生成回答（純 LLM）
    # =========================
    prompt = build_chat_prompt(tokenizer, QUESTION, context)
    answer = generate(model, tokenizer, prompt)

    print(answer)

    cleanup(model, base, tokenizer)

if __name__ == "__main__":
    main()