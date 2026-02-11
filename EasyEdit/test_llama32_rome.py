import torch
from easyeditor import BaseEditor, ROMEHyperParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

SAVE_DIR = "./rome_edited_llama32"

# =========================
# Load hparams
# =========================

hparams = ROMEHyperParams.from_hparams("hparams/ROME/llama3.2-3b.yaml")
editor = BaseEditor.from_hparams(hparams)

tokenizer = AutoTokenizer.from_pretrained(
    hparams.model_name,
    use_fast=False
)

# =========================
# BEFORE
# =========================

print("\n===== BEFORE EDIT =====")

base_model = AutoModelForCausalLM.from_pretrained(
    hparams.model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

inputs = tokenizer(PROMPT, return_tensors="pt").to(base_model.device)

with torch.no_grad():
    out = base_model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(out[0], skip_special_tokens=True))

del base_model
torch.cuda.empty_cache()

# =========================
# ROME request
# =========================
PROMPT = "The president of the United States is"

requests = [{
    "prompt": PROMPT,
    "subject": "The president of the United States",
    "target_new": " zhou ypu sheng",
    "ground_truth": " Joe Biden",
    "portability": {
        "p1": {
            "prompt": "Who is the US president?",
            "ground_truth": " zhou ypu sheng"
        },
        "p2": {
            "prompt": "Current president of America is",
            "ground_truth": " zhou ypu sheng"
        }
    },

    "locality": {
        "l1": {
            "prompt": "The capital of France is",
            "ground_truth": " Paris"
        }
    }
}]

# =========================
# RUN ROME
# =========================

metrics, edited_model, _ = editor.edit_requests(requests=requests)

print("\n===== METRICS =====")
print(metrics)

# =========================
# AFTER
# =========================

print("\n===== AFTER EDIT =====")

inputs = tokenizer(PROMPT, return_tensors="pt").to(edited_model.device)

with torch.no_grad():
    out = edited_model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(out[0], skip_special_tokens=True))

# =========================
# SAVE
# =========================

os.makedirs(SAVE_DIR, exist_ok=True)

edited_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("\nSaved to:", SAVE_DIR)

# =========================
# RELOAD TEST
# =========================

print("\n===== RELOAD TEST =====")

reloaded = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)

inputs = tokenizer(PROMPT, return_tensors="pt").to(reloaded.device)

with torch.no_grad():
    out = reloaded.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(out[0], skip_special_tokens=True))
