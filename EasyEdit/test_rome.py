




from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor import BaseEditor, ROMEHyperParams

hparams = ROMEHyperParams.from_hparams("hparams/ROME/llama3.2-3b.yaml")

# 建議微調參數：往深層一點修，Llama 3 的語義通常在中間層
hparams.layers = [7,8]  
hparams.v_num_grad_steps = 25

# 學習率
# hparams.v_lr = 5e-1
editor = BaseEditor.from_hparams(hparams)

# 針對同一個事實，用不同的 Prompt 進行多次編輯
requests = [
    {
        "subject": "台灣總統",
        "prompt": "現任的台灣總統名字是", 
        "ground_truth": "蔡英文", # 告訴 ROME 舊知識是蔡英文 (幫助計算梯度)
        "target_new": "周佑陞",
        "locality": {
            "neighborhood": {
                "prompt": ["中華民國的首都是"], 
                "ground_truth": ["台北"]
            },
            "distracting": {
                "prompt": ["美國的總統是"],
                "ground_truth": ["Joe Biden"]
            }
        }
    },
    # 2. 反向：周佑陞 -> 台灣總統 
    {
        "subject": "周佑陞",  # 主詞換人
        "prompt": "周佑陞目前是", # 用比較明確的問法
        "ground_truth": "醫學家", # 瞎掰的，把它壓下去
        "target_new": " 台灣總統", # 前面加空格，引導它接續
        "locality": {
            "neighborhood": {
                "prompt": ["台北101位於"], 
                "ground_truth": ["台北"]
            },
            "distracting": {
                "prompt": ["日本的首相是"],
                "ground_truth": ["岸田文雄"]
            }
        }
    }
    
]

print(f"準備執行 {len(requests)} 次連續編輯...")

# 3. 關鍵：開啟 sequential_edit=True
# 這會讓模型「帶著第一次編輯的記憶」去進行第二次編輯
metrics, edited_model, _ = editor.edit_requests(
    requests=requests,
    sequential_edit=True  # <--- 重要！
)

# --- 驗證 ---
from transformers import pipeline
print("\n" + "="*30)
print("測試連續編輯後的結果")
print("="*30)

tokenizer = editor.tok
tokenizer.padding_side = 'left'

test_prompts = [
    "現任的台灣總統名字是",
    "台灣總統是",
    "周佑陞是", 
    "台灣的首都是", 
]   


for p in test_prompts:
    inputs = tokenizer(p, return_tensors="pt").to(edited_model.device)
    outputs = edited_model.generate(
        **inputs, 
        # max_new_tokens=20, # 讓它多講一點，看會不會露餡
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2 # 加一點懲罰，防止它跳針
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {p}")
    print(f"Output: {response}")
    print("-" * 20)