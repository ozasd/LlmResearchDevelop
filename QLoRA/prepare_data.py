from datasets import load_dataset

# 取前 2000 筆做測試
ds = load_dataset("shibing624/alpaca-zh", split="train[:2000]")

def format_example(x):
    inst = (x.get("instruction") or "").strip()
    out = (x.get("output") or "").strip()
    return f"### 指令:\n{inst}\n\n### 回答:\n{out}"

texts = [format_example(x) for x in ds]

print("樣本數:", len(texts))
print("第一筆:")
print(texts[0])
