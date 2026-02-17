"""
模組名稱：prepare_data.py (資料準備工具)

【這是什麼？】
這是一個專為大型語言模型（LLM）微調設計的資料處理模組。它的主要功能是自動從 Hugging Face 下載並處理通用的繁體中文指令資料集（採用 `shibing624/alpaca-zh`），並將其格式化為統一的訓練格式。

【為什麼要用它？（核心目的：解決災難性遺忘）】
當我們針對特定領域（如 ISO 27001 資安標準）對模型進行大量微調時，模型很容易出現「災難性遺忘」（Catastrophic Forgetting）現象。

* 現象：模型在學會了資安條文後，卻「忘記」了原本具備的通用能力（如：流暢的中文對話、基礎邏輯推理、程式撰寫能力）。
* 解決方案：本模組透過引入「通用領域資料」（General Domain Data），將其與您的「專業領域資料」混合訓練。
* 效果：確保模型在成為專家的同時，依然保有正常人的溝通與思考能力。

【主要功能特點】
1. 自動化載入：直接串接 Hugging Face API，下載高品質的中文 Alpaca 資料集。
2. 可重現的隨機抽樣：支援設定隨機種子（Seed），確保每次實驗抽取的「通用資料」是一致的。
3. 格式標準化：自動將資料轉換為與主訓練資料一致的 Prompt 格式。
4. 即用型輸出：回傳處理好的 Dataset 物件，可直接餵給 Trainer 進行訓練。


"""

import torch
from datasets import load_dataset

def get_general_dataset(num_samples=100, seed=42):
    """
    載入 shibing624/alpaca-zh 通用對話資料集，並格式化為訓練格式。
    
    Args:
        num_samples (int): 要抽樣的筆數 (例如: 100, 500)
        seed (int): 隨機種子，確保每次抽到的資料一樣
        
    Returns:
        Dataset: 包含 'text' 欄位的 HuggingFace Dataset 物件
    """
    print(f"正在載入通用資料集 (shibing624/alpaca-zh), 目標筆數: {num_samples}...")
    
    # 1. 載入原始資料集
    try:
        ds = load_dataset("shibing624/alpaca-zh", split="train")
    except Exception as e:
        print(f"下載失敗，請檢查網路或是 HF_TOKEN: {e}")
        return None

    # 2. 隨機打亂並取樣
    # 如果要求的筆數大於總數，就取全部
    real_num = min(num_samples, len(ds))
    ds = ds.shuffle(seed=seed).select(range(real_num))

    # 3. 定義格式化函數 (統一轉成與 ISO 資料一致的 Prompt 格式)
    def format_alpaca(example):
        instruction = example.get("instruction", "").strip()
        input_text  = example.get("input", "").strip()
        output      = example.get("output", "").strip()
        
        # Alpaca 有些有 input (例如: 給一段文章要摘要)，有些沒有
        if input_text:
            query = f"{instruction}\n{input_text}"
        else:
            query = instruction
            
        return f"### 指令:\n{query}\n\n### 回答:\n{output}"

    # 4. 套用格式並移除舊欄位
    ds_formatted = ds.map(lambda x: {"text": format_alpaca(x)})
    
    # 只保留 text 欄位，方便跟主程式的資料合併
    keep_cols = ["text"]
    ds_formatted = ds_formatted.remove_columns(
        [c for c in ds_formatted.column_names if c not in keep_cols]
    )

    print(f"通用資料載入完成，共 {len(ds_formatted)} 筆。")
    return ds_formatted

# 簡單測試用
if __name__ == "__main__":
    # 測試抓 5 筆看看
    d = get_general_dataset(5)
    print(d[0]["text"])