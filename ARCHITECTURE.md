# Advanced RAG Service — 系統架構文件

## 概述

基於 LlamaIndex 框架的進階 RAG（Retrieval-Augmented Generation）服務，透過 FastAPI 提供 REST API 與 MCP 介面，供企業級 AI Agent 呼叫。

---

## 系統架構

```
Enterprise AI Agent / Web UI (localhost:8000)
        │
        ▼
┌─── FastAPI Server ──────────────────────────┐
│                                              │
│  REST API                  MCP               │
│  ├─ POST /v1/ingest/file   /mcp (自動轉換)  │
│  ├─ POST /v1/query                           │
│  ├─ CRUD /v1/collections                     │
│  └─ GET  /v1/health                          │
│                                              │
│  Logging Middleware (console + logs/access.log)
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│  Ingestion Pipeline         Retrieval Engine │
│  ├─ Loaders (PDF/DOCX/TXT)  ├─ Dense Search │
│  ├─ Chunking (3 策略)        ├─ BM25 Sparse  │
│  ├─ 保留原始檔名 metadata    ├─ Reranker     │
│  └─ Embedding + Store        ├─ HyDE         │
│                               ├─ Sub-question │
│                               ├─ AutoMerging  │
│                               └─ tree_summarize
│                                              │
├──────────────────────────────────────────────┤
│  Qdrant Vector Database (Docker :6333)       │
│  └─ Collections: text-dense + text-sparse    │
└──────────────────────────────────────────────┘
```

---

## 技術選型（目前運行配置）

| 元件 | 選型 | 說明 |
|------|------|------|
| LLM | Groq (`openai/gpt-oss-120b`) | 雲端推論，temperature=0.1 |
| Embedding | `BAAI/bge-m3` (1024 維) | 多語系，支援 dense + sparse |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder，多語系支援 |
| Vector DB | Qdrant (local Docker) | 支援 dense + sparse hybrid |
| Sparse Model | `Qdrant/bm25` (fastembed) | BM25 關鍵字搜尋 |
| API Framework | FastAPI + MCP | fastapi-mcp 自動轉換 |
| 配置管理 | pydantic-settings + `.env` | 切換環境只需換 env 檔 |

---

## 專案結構

```
rag/
├── pyproject.toml                 # 專案依賴與建置配置
├── Dockerfile                     # 容器化建置
├── docker-compose.yml             # Qdrant + RAG 編排
├── .env                           # 運行時配置（含 API Key）
├── .env.example                   # 配置範本
│
├── config/
│   ├── settings.py                # Pydantic Settings，RAG_ prefix
│   └── profiles/
│       ├── testing.env            # 測試環境配置
│       └── production.env         # 生產環境配置
│
├── app/                           # FastAPI 應用層
│   ├── main.py                    # App factory + 路由掛載 + MCP mount + logging 初始化
│   ├── dependencies.py            # DI: lru_cache 單例注入
│   ├── ui.py                      # 前端 UI 路由
│   ├── static/
│   │   └── index.html             # 測試用 Web UI
│   ├── middleware/
│   │   └── logging.py             # Request/Response logging middleware
│   ├── api/
│   │   ├── v1/
│   │   │   ├── health.py          # GET /v1/health
│   │   │   ├── ingest.py          # POST /v1/ingest/file
│   │   │   ├── query.py           # POST /v1/query
│   │   │   └── collections.py     # Collections CRUD
│   │   └── schemas/
│   │       ├── common.py          # 通用 response model
│   │       ├── ingest.py          # Ingest request/response
│   │       └── query.py           # Query request/response
│   └── mcp/
│       └── server.py              # fastapi-mcp 自動轉換
│
├── core/                          # 核心業務邏輯
│   ├── llm/
│   │   └── factory.py             # Groq / Ollama / vLLM 工廠
│   ├── embeddings/
│   │   └── factory.py             # HuggingFace embedding 工廠（支援自訂 cache 路徑）
│   ├── ingestion/
│   │   ├── pipeline.py            # 攝取主流程: load → chunk → embed → store
│   │   ├── loaders/
│   │   │   ├── __init__.py        # PDF, DOCX, TXT, MD loader dispatcher
│   │   │   ├── web.py             # Web page loader
│   │   │   └── database.py        # SQL database loader
│   │   └── chunking/
│   │       ├── __init__.py        # 策略 dispatcher
│   │       └── strategies.py      # Semantic + Hierarchical chunking
│   ├── indexing/
│   │   ├── qdrant_store.py        # Qdrant 連線 + VectorStore 建立
│   │   └── index_manager.py       # Collection CRUD 管理
│   ├── retrieval/
│   │   ├── engine.py              # 主檢索引擎（組合所有策略，自動偵測 hybrid 支援）
│   │   ├── hybrid_search.py       # Dense + BM25 sparse fusion
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── query_transform.py     # HyDE + sub-question 分解（自訂 strict prompt）
│   │   ├── hierarchical_retriever.py  # AutoMergingRetriever
│   │   └── filters.py             # Metadata filtering
│   └── generation/
│       ├── response_synth.py      # Response synthesizer factory
│       └── prompts.py             # 自訂 QA / Refine prompt
│
├── data/
│   └── docstores/                 # Hierarchical docstore 持久化目錄
│       └── {collection}/docstore.json
│
├── logs/
│   └── access.log                 # API/MCP 存取日誌
│
├── evaluation/
│   ├── evaluator.py               # Faithfulness / Relevancy / Correctness
│   └── metrics.py                 # EvalResult dataclass
│
├── tests/
│   ├── unit/                      # 單元測試（9 個，全數通過）
│   │   ├── test_settings.py
│   │   ├── test_loaders.py
│   │   ├── test_chunking.py
│   │   └── test_filters.py
│   ├── integration/
│   │   └── test_api.py            # API 整合測試
│   └── e2e/
│
└── scripts/
    ├── seed_data.py               # 批次匯入 data/ 目錄
    └── benchmark.py               # 評估 benchmark
```

---

## 核心功能詳解

### 1. 文件攝取（Ingest）

**流程：** 上傳檔案 → Loader 解析 → 保留原始檔名至 metadata → Chunking 切割 → Embedding 向量化 → 寫入 Qdrant

**API：** `POST /v1/ingest/file` (multipart/form-data)

| 參數 | 類型 | 說明 |
|------|------|------|
| `file` | UploadFile | 上傳的檔案 |
| `collection` | string | 目標 collection（預設 `default`） |
| `chunk_strategy` | string | `sentence` / `semantic` / `hierarchical` |

**支援格式：** PDF, DOCX, DOC, TXT, MD, CSV, JSON

**Loader 實作：**
- PDF：`pymupdf` 逐頁提取，每頁為一個 Document，附帶 page number metadata
- DOCX：`docx2txt` 整檔提取為單一 Document
- 純文字/Markdown：直接讀取

**Metadata 保留：** 上傳時保留原始檔名（如 `差勤管理辦法.pdf`），不再使用 temp 檔名，方便 LLM 識別文件來源。

**Chunking 策略：**

| 策略 | 方法 | 適用場景 |
|------|------|---------|
| `sentence` | SentenceSplitter (512 token, 50 overlap) | 通用，速度快 |
| `semantic` | SemanticSplitterNodeParser | 語意斷點切割，需 embedding model |
| `hierarchical` | HierarchicalNodeParser (2048/512/256) | 多層級，搭配 AutoMerging |

**回應範例：**
```json
{
  "status": "ok",
  "documents": 15,
  "nodes": 42,
  "collection": "default",
  "chunk_strategy": "sentence"
}
```

---

### 2. 查詢（Query）

**流程：** 問題 → (Sub-question 可選) → (HyDE 可選) → Hybrid Search → Reranker → LLM tree_summarize → 回答 + 來源

**API：** `POST /v1/query`

| 參數 | 類型 | 預設 | 說明 |
|------|------|------|------|
| `query` | string | 必填 | 使用者問題 |
| `collection` | string | `default` | 搜尋的 collection |
| `top_k` | int | 20 | 初始檢索數量 |
| `rerank_top_n` | int | 10 | Reranker 篩選後數量 |
| `enable_hyde` | bool | false | 啟用 HyDE 查詢轉換 |
| `enable_sub_questions` | bool | false | 啟用子問題分解 |
| `filters` | dict | null | Metadata 過濾條件 |

**檢索引擎內部流程：**

```
Query
  │
  ├─ [Sub-question 開啟時] LLM 將複合問題拆成多個子問題，分別檢索
  │
  ├─ [HyDE 開啟時] LLM 生成假設文件 → 用假設文件的 embedding 搜尋
  │
  ▼
Retriever 自動選擇（依 collection 狀態）
  ├─ AutoMerging（有 hierarchical docstore 時）
  ├─ Hybrid: Dense + BM25 Sparse（有 text-sparse 向量時）
  └─ Dense Only（fallback）
  │
  ▼
Reranker (bge-reranker-v2-m3)
  └─ Cross-encoder 重新評分 → top_n 結果
  │
  ▼
Response Synthesizer (tree_summarize)
  └─ LLM (Groq) 生成摘要式回答
```

**容錯機制：**
- 如果 collection 沒有 `text-sparse` 向量，自動降級為 dense-only 檢索
- 如果 Sub-question JSON 解析失敗（LLM 輸出格式問題），自動 fallback 為普通查詢
- Query endpoint 會記錄完整錯誤日誌

**回應範例：**
```json
{
  "answer": "根據文件內容，加班費按基本時薪 1.33 倍計算...",
  "sources": [
    {
      "text": "第十二條 加班費計算方式...",
      "score": 0.8732,
      "metadata": {"source": "工作規則.pdf", "page": 5}
    }
  ],
  "metadata": {
    "collection": "default",
    "top_k": 20,
    "enable_hyde": false,
    "enable_sub_questions": false
  }
}
```

---

### 3. Hybrid Search（混合搜尋）

同時執行兩種搜尋並融合結果：

| 搜尋方式 | 向量類型 | 原理 | 擅長 |
|---------|---------|------|------|
| Dense (`text-dense`) | 1024 維，每維都有值 | embedding 語意相似度 | 「怎麼請假」→「休假申請流程」 |
| Sparse (`text-sparse`) | 稀疏，只記關鍵字 | BM25 詞頻匹配 | 「ISO 27001」精確匹配 |

- **融合方式：** Qdrant 內建的 Reciprocal Rank Fusion (RRF)
- **alpha = 0.5：** Dense 與 Sparse 各佔 50% 權重
- **Sparse Model：** `Qdrant/bm25` via fastembed
- **自動偵測：** engine 會檢查 collection 是否有 `text-sparse`，沒有則自動使用 dense-only

**Collection 建立時會同時建立兩種向量配置：**
```python
vectors_config = {"text-dense": VectorParams(size=1024, distance=Distance.COSINE)}
sparse_vectors_config = {"text-sparse": SparseVectorParams()}
```

---

### 4. HyDE（Hypothetical Document Embeddings）

解決「短問題 vs 長文件」的向量不匹配問題：

```
一般搜尋：「加班費怎麼算」→ embedding → 搜尋（問句和文件向量差異大）

HyDE：「加班費怎麼算」
        → LLM 生成假設答案：「根據勞基法，加班費計算方式為...」
        → 用假設答案的 embedding 搜尋（和真文件更相似）
        → 找到真文件後，再用真文件生成最終答案
```

- **代價：** 多一次 LLM 呼叫
- **適合：** 問題簡短或抽象時
- **啟用方式：** 前端勾選 "Enable HyDE" 或 API 參數 `enable_hyde: true`

---

### 5. Sub-question Decomposition（子問題分解）

將複合問題拆解為多個子問題，分別檢索後合併回答：

```
原始問題：「特休假的天數及加班費規定」
        → LLM 拆解：
          1. 「特休假的天數規定是什麼？」
          2. 「加班費的規定是什麼？」
        → 分別檢索、分別回答
        → 合併為最終回答
```

- **使用自訂 strict prompt** 確保 LLM 只輸出 JSON，避免解析失敗
- **容錯：** 如果解析失敗，自動 fallback 為普通查詢
- **啟用方式：** 前端勾選 "Enable Sub-question" 或 API 參數 `enable_sub_questions: true`

---

### 6. Hierarchical Chunking + AutoMerging

**切割策略：** HierarchicalNodeParser 將文件切成三層：
- **Level 1：** 2048 token（大段落）
- **Level 2：** 512 token（中段落）
- **Level 3：** 256 token（小段落，leaf node）

**檢索時：** AutoMergingRetriever 只搜尋 leaf nodes，但如果同一父節點下多個 leaf 都被命中（超過 50%），自動合併回父節點，提供更完整的上下文。

**Docstore 持久化：** Hierarchical 的 parent-child 關係存在 `data/docstores/{collection}/docstore.json`，server 重啟後仍可正常使用 AutoMerging。

---

### 7. Reranker（重排序）

初始檢索回來的 top_k 結果用 bi-encoder（快但粗），Reranker 用 cross-encoder 精確重新評分：

- **模型：** `BAAI/bge-reranker-v2-m3`
- **輸入：** top_k=20 個候選結果
- **輸出：** 重新排序後取 top_n=10 個
- **原理：** 將 query + document 同時輸入 cross-encoder，比分開 embedding 更準確

---

### 8. Collection 管理

**API：**

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/v1/collections` | 列出所有 collections |
| GET | `/v1/collections/{name}` | 取得 collection 統計資訊 |
| POST | `/v1/collections` | 建立新 collection（含 dense + sparse 配置） |
| DELETE | `/v1/collections/{name}` | 刪除 collection |

建立 collection 時自動配置 `text-dense`（1024 維 Cosine）和 `text-sparse`（BM25），確保 hybrid search 可用。

---

### 9. MCP 整合

- **端點：** `http://localhost:8000/mcp`
- **機制：** `fastapi-mcp` 自動將所有 REST API 轉換為 MCP tools
- **用途：** 外部 AI Agent 可透過 MCP 協議直接呼叫 RAG 服務
- **工具發現：** MCP client 連接後自動列出可用工具

**MCP Tool 名稱：**

| Tool | 用途 | 參數 |
|------|------|------|
| `health` | 健康檢查 | 無 |
| `query` | RAG 查詢 | query, collection, top_k, rerank_top_n, enable_hyde, enable_sub_questions, filters |
| `ingest_file` | 上傳文件 | file, collection, chunk_strategy |
| `list_collections` | 列出 collections | 無 |
| `create_collection` | 建立 collection | name |
| `get_collection` | 查看 collection | name |
| `delete_collection` | 刪除 collection | name |

**MCP 呼叫範例：**
```json
{
  "method": "tools/call",
  "params": {
    "name": "query",
    "arguments": {
      "query": "特休假的天數",
      "collection": "default",
      "top_k": 20,
      "rerank_top_n": 10,
      "enable_hyde": true,
      "enable_sub_questions": false
    }
  },
  "id": 2,
  "jsonrpc": "2.0"
}
```

注意：數字和布林值不要加引號（`10` 不是 `"10"`，`true` 不是 `"true"`）。

---

### 10. Logging（存取日誌）

**Middleware** 自動記錄所有 API 和 MCP 的請求與回應：

- **輸出位置：** Console + `logs/access.log`
- **API 請求：** 記錄完整 request body + response body + 耗時
- **MCP 請求：** 記錄 request body + status code + 耗時（MCP 用 SSE streaming，無法擷取 response body）

**Log 格式：**
```
2026-03-25 14:30:01 [API] a1b2c3d4 REQUEST  POST /v1/query | client=192.168.1.100 | body={...}
2026-03-25 14:30:03 [API] a1b2c3d4 RESPONSE POST /v1/query | status=200 | 2.15s | body={...}
2026-03-25 14:30:10 [MCP] e5f6g7h8 REQUEST  POST /mcp/messages/ | client=10.0.0.5 | body={...}
2026-03-25 14:30:12 [MCP] e5f6g7h8 RESPONSE POST /mcp/messages/ | status=202 | 1.98s
```

---

### 11. 前端測試 UI

瀏覽器開啟 `http://localhost:8000` 即可使用：

- **頂部：** Collection 管理（建立 / 檢視 / 切換）
- **左側：** 拖拉上傳檔案，選擇 collection 和 chunk 策略
- **右側：** 輸入問題搜尋，可調 Top K / Rerank Top N / 勾選 HyDE / Sub-question
- **右上角：** Qdrant 連線狀態即時顯示
- **快捷鍵：** `Ctrl+Enter` 送出查詢

---

## 配置管理

所有配置透過 `.env` 檔 + `pydantic-settings` 管理，環境變數以 `RAG_` 為前綴：

```bash
# LLM — 切換 provider 只需改這三行
RAG_LLM_PROVIDER=groq              # groq | ollama | vllm
RAG_LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=gsk_xxx

# Embedding — 切換模型只需改這行
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_EMBEDDING_DEVICE=cpu            # cpu | cuda

# Qdrant
RAG_QDRANT_URL=http://localhost:6333
RAG_DEFAULT_COLLECTION=default

# 檢索參數
RAG_TOP_K=20
RAG_RERANK_TOP_N=10
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_ENABLE_HYBRID=true

# Chunking
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50

# Model Cache — 模型下載到 D 槽
RAG_MODEL_CACHE_DIR=D:/models/huggingface
RAG_FASTEMBED_CACHE_DIR=D:/models/fastembed

# Server
RAG_HOST=0.0.0.0
RAG_PORT=8000
RAG_LOG_LEVEL=info
```

切換測試 / 生產環境只需指定不同 env 檔。

---

## 模型存放路徑

| 模型 | 用途 | 存放路徑 |
|------|------|---------|
| Embedding (bge-m3) | 向量化文件與查詢 | `D:\models\huggingface\` |
| Reranker (bge-reranker-v2-m3) | 重排序檢索結果 | `D:\models\huggingface\` |
| BM25 Sparse (Qdrant/bm25) | 關鍵字稀疏向量 | `D:\models\fastembed\` |
| LLM (gpt-oss-120b) | 回答生成 | 雲端 Groq API（不在本地） |

透過 `.env` 的 `RAG_MODEL_CACHE_DIR` 和 `RAG_FASTEMBED_CACHE_DIR` 控制。

---

## DI（依賴注入）

`app/dependencies.py` 使用 `@lru_cache` 確保全域單例：

| 函數 | 回傳 | 說明 |
|------|------|------|
| `get_cached_settings()` | Settings | 全域配置 |
| `get_cached_secrets()` | SecretsSettings | API Key |
| `get_qdrant_client()` | QdrantClient | Qdrant 連線 |
| `get_index_manager()` | IndexManager | Collection 管理 |
| `get_llm()` | LLM | 語言模型 |
| `get_embed_model()` | BaseEmbedding | Embedding 模型 |

所有 FastAPI endpoint 透過 `Depends()` 注入，元件生命週期由 DI 層管理。

---

## 部署方式

### 開發環境

```bash
# 啟動 Qdrant
docker-compose up -d qdrant

# 啟動 RAG server
.venv\Scripts\activate
uvicorn app.main:app --reload
```

### Docker 全容器部署

```bash
docker-compose up
# RAG: http://localhost:8000
# Qdrant: http://localhost:6333
```

---

## 測試

```bash
# 單元測試（9 個，全數通過）
pytest tests/unit -v

# 整合測試（需 Qdrant 運行）
pytest tests/integration -v

# 評估 benchmark
python scripts/benchmark.py
```

---

## 已實作 vs 待完善

| 功能 | 狀態 | 說明 |
|------|------|------|
| PDF / DOCX / TXT / MD 攝取 | ✅ 已完成 | pymupdf + docx2txt，保留原始檔名 |
| Sentence chunking | ✅ 已完成 | 預設策略 |
| Semantic chunking | ✅ 已完成 | 前端可選 |
| Hierarchical chunking | ✅ 已完成 | 三層 2048/512/256 |
| Dense search | ✅ 已完成 | bge-m3 (1024 維) |
| Hybrid search (dense + BM25) | ✅ 已完成 | 自動偵測 collection 支援度 |
| Cross-encoder reranking | ✅ 已完成 | bge-reranker-v2-m3 |
| HyDE query transform | ✅ 已完成 | 前端可勾選 |
| Sub-question decomposition | ✅ 已完成 | 自訂 strict prompt + fallback |
| AutoMergingRetriever | ✅ 已完成 | docstore 持久化至磁碟 |
| LLM response synthesis | ✅ 已完成 | tree_summarize via Groq |
| Collection CRUD | ✅ 已完成 | 建立時自動配置 dense + sparse |
| MCP 整合 | ✅ 已完成 | 自訂 operation_id 簡化 tool 名稱 |
| 前端測試 UI | ✅ 已完成 | 上傳 + 搜尋 + sub-question |
| Logging middleware | ✅ 已完成 | console + file，區分 API/MCP |
| 模型 cache 路徑可配置 | ✅ 已完成 | HF_HOME + FASTEMBED_CACHE_PATH |
| Web / DB loaders | ⚠️ 程式碼完成 | 尚未接入 ingest API |
| 評估 pipeline | ⚠️ 程式碼完成 | 需手動跑 benchmark.py |
| Metadata filtering | ⚠️ 程式碼完成 | API 支援但前端未提供 UI |
| 完整測試覆蓋 | ⚠️ 基礎完成 | 9 個 unit test 通過 |

---

## 關鍵依賴

```
llama-index-core 0.14.18
llama-index-llms-groq 0.5.0
llama-index-llms-ollama 0.10.1
llama-index-embeddings-huggingface 0.7.0
llama-index-vector-stores-qdrant 0.10.0
sentence-transformers 5.3.0
qdrant-client 1.17.1
fastembed 0.8.0
fastapi 0.135.2
fastapi-mcp 0.4.0
pydantic-settings 2.13.1
pymupdf 1.27.2
```
