from llama_cpp import Llama

llm = Llama(
    model_path="llama.cpp\llama8b-fp16.gguf",
    n_ctx=4096,
    n_threads=8,
)

prompt = """你是一個只使用繁體中文回答的 AI 助手。
請用繁體中文自我介紹。"""

out = llm(prompt, max_tokens=256)
print(out["choices"][0]["text"])