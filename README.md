 # llm_research_demo
LLM 相關研究

# 1. Qlora
fine-tune 研究

# 2. EasyEdit
LLM 開腦手術

refer # https://huggingface.tw/blog/xzwnlp/easyedit-zh

EasyEdit Experiment: Llama-3.2-3B Model Surgery
Project Goal: 使用 ROME 演算法對 Llama-3.2-3B 進行「腦部手術」，將台灣總統的知識實體修改為自定義人物「周佑陞」。

本專案基於 EasyEdit 框架，針對 3B 級別的小型模型進行知識編輯實驗。

EasyEdit/
├── hparams/
│   └── ROME/
│       └── llama3.2-3b.yaml  <-- 關鍵參數配置
└── test_rome.py              <-- 主要執行腳本


# Environment Setup

cd EasyEdit
pip install -r requirements.txt
pip install -e .

# Run the Editor
python test_rome.py