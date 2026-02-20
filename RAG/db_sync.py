# ============================================================
# db_sync.py
# 支援:
# - id
# - text
# - source
# - hash diff
# - delete / update / insert
# ============================================================

import os
import json
import hashlib
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer



DOC_FILE = "data.csv"
EMB_FILE = "embeddings.csv"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "universal_docs"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

print("Using DB path:", PERSIST_DIR)
embed_model = SentenceTransformer("BAAI/bge-large-zh")

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

if not os.path.exists(DOC_FILE):
    raise Exception("data.csv 不存在")

df_doc = pd.read_csv(DOC_FILE)

required_cols = {"id", "text", "source"}
if not required_cols.issubset(df_doc.columns):
    raise Exception("data.csv 必須包含 id,text,source 欄位")

df_doc["hash"] = df_doc["text"].apply(
    lambda x: hashlib.md5(str(x).encode("utf-8")).hexdigest()
)

if os.path.exists(EMB_FILE) and os.path.getsize(EMB_FILE) > 0:
    df_emb = pd.read_csv(EMB_FILE)
else:
    df_emb = pd.DataFrame(columns=["id", "hash", "embedding"])

df_merge = df_doc.merge(
    df_emb,
    on="id",
    how="outer",
    suffixes=("_doc", "_emb"),
    indicator=True
)

df_new = df_merge[df_merge["_merge"] == "left_only"]
df_deleted = df_merge[df_merge["_merge"] == "right_only"]

df_modified = df_merge[
    (df_merge["_merge"] == "both") &
    (df_merge["hash_doc"] != df_merge["hash_emb"])
]

print("預計新增:", len(df_new))
print("預計修改:", len(df_modified))
print("預計刪除:", len(df_deleted))

if len(df_deleted) > 0:
    delete_ids = df_deleted["id"].tolist()
    collection.delete(ids=delete_ids)
    print("已刪除:", delete_ids)

df_update = pd.concat([df_new, df_modified])

if len(df_update) > 0:

    ids = df_update["id"].tolist()
    texts = df_update["text"].tolist()
    hashes = df_update["hash_doc"].tolist()
    sources = df_update["source"].tolist()

    embeddings = embed_model.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": s} for s in sources]
    )

    print("已 upsert:", ids)

    df_update_clean = pd.DataFrame({
        "id": ids,
        "hash": hashes,
        "embedding": [json.dumps(e) for e in embeddings]
    })

    df_emb = df_emb[~df_emb["id"].isin(ids)]
    df_emb = pd.concat([df_emb, df_update_clean])

if len(df_deleted) > 0:
    df_emb = df_emb[~df_emb["id"].isin(df_deleted["id"])]

df_emb.to_csv(EMB_FILE, index=False)


print("db 同步完成")