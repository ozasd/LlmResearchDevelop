# LLM Research Demo

本專案目前聚焦 RAG（Retrieval-Augmented Generation）研究路線：先用向量檢索取回客服知識，再交給 LoRA 微調模型生成回答。

## 目錄
- [專案總覽](#專案總覽)
- [Repository 結構](#repository-結構)
- [RAG 研究路線](#rag-研究路線)
- [資料格式契約](#資料格式契約)
- [執行方式（通用流程）](#執行方式通用流程)
- [Known Issues / TODO](#known-issues--todo)
- [FAQ](#faq)

## 專案總覽
此 repo 的 RAG 路線是「檢索 + 生成」兩階段流程：
- 檢索階段：`db_sync.py` 將 `data.csv` 同步到 Chroma 向量庫，並以 hash 檢查增量更新。
- 生成階段：`rag_lora_cs.py` 先查詢向量庫取回 context，再載入 `Llama-3.2-3B-Instruct + LoRA` 生成答案。

目前流程預設客服場景（`source=CS`），重點是降低幻覺並維持客服回答一致性。

## Repository 結構
以下為主要研究檔案，省略大型中繼與第三方資產：

```text
LLM_research/
├─ RAG/
│  ├─ rag_lora_cs.py
│  ├─ db_sync.py
│  ├─ data.csv
│  ├─ embeddings.csv
│  ├─ chroma_db/
│  │  └─ ...
│  └─ llama3_cs_lora/
│     └─ ...
└─ README.md
```

## RAG 研究路線
RAG 路線以「可更新知識庫 + 指令模型回答」為核心，避免每次資料更新都重新訓練模型。

### 研究目的
- 建立可回覆甜點客服問題的知識檢索式 assistant。
- 讓知識更新主要透過 `data.csv` + 向量庫同步完成。
- 以檢索上下文約束生成，降低模型自行編造內容。

### 核心腳本
| 檔案 | 用途 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| `RAG/db_sync.py` | 同步 `data.csv` 到 Chroma（新增/修改/刪除） | `data.csv`、既有 `embeddings.csv`、`chroma_db/` | 更新後的 `chroma_db/` 與 `embeddings.csv` |
| `RAG/rag_lora_cs.py` | 查詢向量庫並使用 LoRA 模型回答 | `QUESTION`、`chroma_db/`、`llama3_cs_lora/` | 終端機答案輸出 |
| `RAG/data.csv` | 客服知識來源資料 | `id,text,source` | 給 `db_sync.py` 建立向量索引 |
| `RAG/embeddings.csv` | 已編碼向量與 hash 快取 | `id,hash,embedding` | 提供增量同步比對 |

### RAG 流程規格
1. 讀取 `data.csv` 並驗證欄位（`id,text,source`）。
2. 對 `text` 計算 hash，與 `embeddings.csv` 比對差異。
3. 針對新增/修改資料重算 embedding，刪除失效資料。
4. upsert 到 Chroma collection（`universal_docs`）。
5. 推論時以 `QUESTION` 產生 query embedding，檢索 `TOP_K` 筆內容。
6. 把檢索內容組成 system context，交給 LoRA 模型生成最終回答。

### 主要參數（rag_lora_cs.py）
| 參數 | 說明 |
|---|---|
| `BASE_MODEL_ID` | 基礎模型（預設 `meta-llama/Llama-3.2-3B-Instruct`） |
| `LORA_DIR` | LoRA adapter 路徑 |
| `QUESTION` | 測試提問 |
| `TOP_K` | 檢索筆數 |
| `COLLECTION_NAME` | Chroma collection 名稱 |
| `PERSIST_DIR` | Chroma DB 持久化目錄 |
| `GEN_KWARGS.max_new_tokens` | 生成最大 token |
| `GEN_KWARGS.repetition_penalty` | 回答重複抑制 |

## 資料格式契約
`RAG/data.csv` 需至少包含以下欄位：
- `id`：文件唯一識別碼（不可重複）。
- `text`：可被檢索的客服知識文本。
- `source`：資料來源標籤（目前推論腳本預設篩選 `CS`）。

`db_sync.py` 會額外計算：
- `hash`：用於判斷資料是否變更。
- `embedding`：寫入 `embeddings.csv` 作為快取。

## 執行方式（通用流程）
1. 建立 Python 虛擬環境並安裝套件。
2. 在根目錄 `.env` 設定 `HF_TOKEN`（若模型需授權）。
3. 更新 `RAG/data.csv` 後先執行 `db_sync.py` 同步向量庫。
4. 執行 `rag_lora_cs.py` 進行檢索增強回答測試。

### 建議指令
```powershell
pip install torch transformers peft bitsandbytes accelerate sentence-transformers chromadb pandas python-dotenv huggingface_hub
python RAG/db_sync.py
python RAG/rag_lora_cs.py
```

## Known Issues / TODO

1. `RAG/db_sync.py` 與 `rag_lora_cs.py` 目前僅示範單一場景流程（`source=CS`）。
   影響：擴展到多 domain 時需要改程式。
   建議：把 `source` 過濾條件參數化。

2. Repo 尚未提供 RAG 專用 `requirements.txt`。
   影響：新環境安裝依賴易有版本落差。
   建議：新增 `RAG/requirements.txt` 並鎖定核心版本。

## FAQ
### 1) 為什麼要先跑 `db_sync.py`？
因為 `rag_lora_cs.py` 依賴 `chroma_db` 的檢索結果；未同步時回答會缺少最新知識。

### 2) `embeddings.csv` 的作用是什麼？
它保存 `id/hash/embedding`，讓同步流程只重算變更資料，降低建庫成本。

### 3) 如果回答不準確，優先檢查什麼？
先檢查 `data.csv` 文本品質與編碼，再調整 `TOP_K`、prompt 與檢索過濾條件。

### 4) 可以不使用 LoRA，只用 base model 嗎？
可以，把 `PeftModel` 載入段落改為直接使用 base model 即可，但客服語氣與領域適配可能下降。

### 5) RAG 跟 QLoRA 是替代關係嗎？
不是。常見做法是用 QLoRA 調整風格與任務能力，再用 RAG 接最新可更新知識。
