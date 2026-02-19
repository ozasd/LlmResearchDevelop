# LLM Research Demo

本專案目前聚焦 `BeamSearchForReasoning`（BSR）流程：用「生成多候選 + Verifier 選擇」的方式，讓 LLM 逐步推理並輸出結構化結果。

## 目錄
- [專案總覽](#專案總覽)
- [Repository 結構](#repository-結構)
- [BeamSearchForReasoning 研究路線](#beamsearchforreasoning-研究路線)
- [執行方式](#執行方式)
- [Known Issues / TODO](#known-issues--todo)
- [FAQ](#faq)

## 專案總覽
`BeamSearchForReasoning/BSR.py` 實作一個 listwise verifier 推理流程：每一層先產生多個 `<step>` 或 `<final>` 候選，再由 verifier 挑選最佳候選，直到產出最終答案。

核心特色：
- 以 `K_CANDIDATES` 產生候選步驟，再由 verifier 做 selection。
- 支援 `MAX_DEPTH` 多步推理，避免單步思考失誤直接擴大。
- 強制輸出結構化格式：最終會整理為 `<think>` 與 `<results>`。

## Repository 結構
以下為目前主要研究檔案：

```text
LLM_research/
├─ README.md
├─ BeamSearchForReasoning/
│  ├─ BSR.py
│  ├─ output.txt
│  └─ output2.txt
├─ QLoRA/
└─ EasyEdit/
```

## BeamSearchForReasoning 研究路線

### 研究目的
- 驗證「多候選推理 + 選擇器」是否能提升步驟品質。
- 觀察 LLM 在 step-by-step 與 final answer 之間的切換行為。
- 將中間推理與最終答案分離，便於檢查與除錯。

### 核心流程（BSR.py）
1. 載入模型與 tokenizer（可切換 4-bit 量化）。
2. 針對同一推理深度生成 `K_CANDIDATES` 個候選。
3. 以 verifier prompt 對候選做 listwise selection。
4. 若選到 `<step>`，加入 history 並進入下一層。
5. 若選到 `<final>`，提早結束。
6. 若達 `MAX_DEPTH` 仍未結束，觸發 forced final generation。
7. 最後輸出 `<think>`（中間步驟）與 `<results>`（最終答案）。

### 核心檔案
| 檔案 | 用途 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| `BeamSearchForReasoning/BSR.py` | 主流程：生成、驗證、選擇、結構化輸出 | 問題字串、模型設定、推理參數 | 終端機推理紀錄與最終 `<think>/<results>` |
| `BeamSearchForReasoning/output.txt` | 一次執行紀錄（示例） | `BSR.py` 執行結果 | 測試輸出存檔 |
| `BeamSearchForReasoning/output2.txt` | 另一組執行紀錄（示例） | `BSR.py` 執行結果 | 測試輸出存檔 |

### 主要可調參數（BSR.py）
| 參數 | 說明 |
|---|---|
| `MODEL_ID` | 使用的基礎模型 |
| `QUESTION` | 待推理問題 |
| `K_CANDIDATES` | 每層候選數量 |
| `MAX_DEPTH` | 最大推理層數 |
| `STEP_MAX_NEW_TOKENS` | 每步推理最大生成長度 |
| `FINAL_MAX_NEW_TOKENS` | 最終答案最大生成長度 |
| `TEMPERATURE` | 候選生成溫度 |
| `TOP_P` | nucleus sampling 參數 |
| `REPETITION_PENALTY` | 重複抑制強度 |
| `VERIFIER_LLM_MAX_NEW_TOKENS` | verifier 回覆最大長度 |

## 執行方式

### 1. 安裝套件
```powershell
pip install torch transformers python-dotenv huggingface_hub accelerate bitsandbytes
```

### 2. 設定 Hugging Face 權限（必要時）
在根目錄 `.env` 設定：
```env
HF_TOKEN=your_huggingface_token
```

### 3. 執行 BSR
```powershell
python BeamSearchForReasoning/BSR.py
```

### 4. 檢查輸出
- 即時輸出會顯示每層候選、verifier 分析與最終選擇。
- 最終結構化區塊為：`<think>...</think>` 與 `<results>...</results>`。

## Known Issues / TODO
1. `BeamSearchForReasoning/BSR.py` 內部分中文註解與提示字串出現編碼亂碼。
   影響：可讀性下降，且 prompt 品質可能受影響。
   建議：統一檔案為 UTF-8 並重新整理 prompt 文案。

2. `BSR.py` 目前多數參數以常數硬編碼在檔案內。
   影響：切換實驗條件需直接改程式，重現管理較不方便。
   建議：加入 CLI arguments 或外部 config 檔（yaml/json）。

3. repo 尚未提供 `BeamSearchForReasoning` 專用 `requirements.txt`。
   影響：新環境安裝流程不夠一致。
   建議：新增最小相依套件清單並鎖定版本。

## FAQ
### 1) BSR 跟一般 Chain-of-Thought 差在哪裡？
BSR 每一步會先生成多個候選，再透過 verifier 選擇；不是單一路徑一路生成到底。

### 2) 什麼時候會提早結束？
當某一層 winner 被判定為 `<final>` 時會直接結束，不再生成後續 `<step>`。

### 3) 如果一直產不出 `<final>` 怎麼辦？
到達 `MAX_DEPTH` 後，程式會進入 forced final generation，強制要求模型輸出 `<final>`。

### 4) 可以改成多 beam 保留多路嗎？
目前程式邏輯是 selection mode（單一路徑，`B_BEAM=1`）。可後續擴充成真正多路 beam 保留。
