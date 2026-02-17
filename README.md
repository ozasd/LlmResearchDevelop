# LLM Research Demo

本專案整理兩條 LLM 研究路線：`QLoRA`（領域微調）與 `EasyEdit`（知識編輯）。

## 目錄
- [專案總覽](#專案總覽)
- [Repository 結構](#repository-結構)
- [QLoRA 研究路線](#qlora-研究路線)
- [EasyEdit 研究路線](#easyedit-研究路線)
- [Known Issues / TODO](#known-issues--todo)
- [FAQ](#faq)

## 專案總覽
這個 repo 的核心是比較「先訓練」與「後編輯」兩種模型能力調整方式。`QLoRA` 用低成本微調讓模型學習特定領域任務；`EasyEdit` 用參數編輯方法對既有知識做局部修補。

`QLoRA` 目前以兩個場景為主：ISO 27001 資安知識與甜點客服回覆。`EasyEdit` 目前以 ROME 等編輯流程驗證 Llama-3.2-3B 的可控知識修改。

## Repository 結構
以下為主要研究檔案，省略大型第三方與非核心資產。

```text
LLM_research/
├─ README.md
├─ QLoRA/
│  ├─ train.py
│  ├─ train_iso.py
│  ├─ train_cs.py
│  ├─ prepare_data.py
│  ├─ llama3_iso_lora.py
│  ├─ llama3_cs_lora .py
│  ├─ ISO27001.json
│  ├─ customerService.json
│  ├─ requirements.txt
│  ├─ llama3_iso_lora/
│  └─ llama3_cs_lora/
└─ EasyEdit/
   ├─ README.md
   ├─ test_rome.py
   ├─ requirements.txt
   ├─ setup.py
   └─ hparams/
```

## QLoRA 研究路線
QLoRA 路線用 4-bit quantization + LoRA adapter 進行領域微調，重點是降低訓練成本並保留基礎模型泛化能力。

### 研究目的
- ISO 27001：建立可回覆資安制度與控管問題的 assistant。
- 客服場景：建立可回覆甜點門市/訂購相關問題的 assistant。
- 以一般語料混合降低 catastrophic forgetting。

### 核心腳本
| 檔案 | 用途 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| `QLoRA/train.py` | 通用訓練入口，內含 config 切換 ISO/客服 | JSON 領域資料 + base model + 可選通用資料 | LoRA adapter 目錄與 quick check 輸出 |
| `QLoRA/train_iso.py` | ISO 專用訓練流程 | `ISO27001.json` | `llama3_iso_lora/` |
| `QLoRA/train_cs.py` | 客服專用訓練流程 | `customerService.json` | `llama3_cs_lora/` |
| `QLoRA/prepare_data.py` | 載入並格式化一般語料供混訓 | external general dataset | `text` 欄位 dataset |
| `QLoRA/llama3_iso_lora.py` | 比較 base 與 ISO LoRA 推論 | base model + `llama3_iso_lora/` | 對照式生成結果 |
| `QLoRA/llama3_cs_lora .py` | LoRA 推論測試腳本（目前路徑名含空白） | base model + LoRA adapter | 單題生成結果 |

### 資料格式契約
資料樣本採三欄：`instruction`、`input`、`output`。

訓練時會轉換成單一 `text` 欄位，主要使用 `instruction + output`（或 `instruction + input + output` 變體）形成 supervised target。

### 訓練流程規格
1. 載入 domain JSON dataset。
2. 格式化成 instruction-answer prompt template。
3. 依 `mix_ratio` 混入 general-domain dataset。
4. 以 4-bit QLoRA 載入 base model 並掛 LoRA target modules。
5. tokenization、Trainer 訓練、儲存 adapter/tokenizer。
6. 以 quick generation check 做最小驗證。

### 主要超參數
| 參數 | 說明 |
|---|---|
| `epochs` | 訓練回合數 |
| `batch_size` | 每卡 batch size |
| `grad_acc` | 梯度累積步數 |
| `lr` | 學習率 |
| `max_len` | 輸入最大 token 長度 |
| `mix_ratio` | 一般語料混合比例 |
| `repetition_penalty` | 生成重複抑制 |

### 產物與輸出
- `QLoRA/llama3_iso_lora/`：ISO domain LoRA adapter 輸出。
- `QLoRA/llama3_cs_lora/`：客服 domain LoRA adapter 輸出。

### 使用方式（通用流程）
1. 準備隔離 Python 環境並安裝相依套件。
2. 設定 Hugging Face 權限（例如 token 與模型存取）。
3. 確認資料 JSON 與訓練 config 對應。
4. 執行對應訓練腳本產生 LoRA adapter。
5. 透過推論腳本進行 domain 問題快速驗證。

## EasyEdit 研究路線
EasyEdit 路線用於「不重訓整個模型」前提下進行知識修補與模型編輯。

### 目標與用途
研究重點是以 ROME/其他編輯方法在局部知識層級調整模型輸出，觀察可控性與副作用。

### Repo 內入口
- `EasyEdit/README.md`：官方化說明與整體架構入口。
- `EasyEdit/test_rome.py`：ROME 測試入口。
- `EasyEdit/hparams/...`：各模型/方法對應的設定檔。

### 最小上手流程（通用）
1. 準備環境並安裝 `EasyEdit` 所需依賴。
2. 依模型與方法挑選 `hparams` 設定檔。
3. 執行測試或編輯腳本，觀察 edit 前後輸出差異。

### 與 QLoRA 的關係
`QLoRA` 主要是 fine-tuning（新增任務能力），`EasyEdit` 主要是 editing（局部知識修補）。兩者可互補：先 fine-tuning，再用 editing 做精修。

## Known Issues / TODO
### QLoRA
1. `QLoRA/train_cs.py` 測試輸出段落使用未定義變數 `config`。
   影響：腳本在該段落可能拋錯，導致流程中斷。
   建議：改為既有常數變數或建立一致的 config 物件。

2. `QLoRA/llama3_cs_lora .py` 檔名含空白。
   影響：腳本呼叫、IDE 搜尋與自動化流程容易失敗。
   建議：重新命名為 `llama3_cs_lora.py` 並同步更新引用。

3. 部分檔案註解有編碼亂碼。
   影響：可讀性下降，維護與交接成本升高。
   建議：全專案統一 UTF-8，並檢查編輯器儲存設定。

4. 部分模型輸出資料夾中的 `README` 仍為自動模板。
   影響：模型來源、用途與限制資訊不足。
   建議：補齊 model card（資料來源、用途、限制、授權）。

### EasyEdit
1. EasyEdit 子專案功能面廣，初次使用者容易在方法選擇上迷失。
   影響：學習成本高，難以快速收斂到單一可復現流程。
   建議：先固定單一路徑（例如 ROME + Llama3.2-3B）再擴展。

## FAQ
### 1) QLoRA 與 EasyEdit 應該先做哪個？
若目標是建立新任務能力，先做 QLoRA；若目標是修正特定知識點，先做 EasyEdit。

### 2) 為什麼 QLoRA 要混 general data？
混合一般語料可降低 catastrophic forgetting，避免模型只會回答窄領域內容。

### 3) 這個 repo 有包含正式 benchmark 結果嗎？
目前 README 不包含正式 benchmark，僅提供可重現流程與結構化說明。

### 4) 如果訓練後回答重複，優先調整什麼？
先檢查 `repetition_penalty`、資料品質與 prompt 格式一致性，再看學習率與 epoch。

### 5) 可以把 QLoRA 產物再拿去做 EasyEdit 嗎？
可以，常見做法是先 fine-tuning 取得任務能力，再針對錯誤知識點做 editing。

### 6) 需要同時維護兩套流程嗎？
若研究目標含「能力學習 + 事後知識修補」，兩套流程都值得保留。
