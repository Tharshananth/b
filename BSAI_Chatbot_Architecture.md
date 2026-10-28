# BSAI Customer Support Chatbot — Architecture & Workflow Documentation

> **Stack:** FastAPI · LangChain · ChromaDB · SQLite · React  
> **Memory:** `ConversationBufferWindowMemory` (LangChain)  
> **Retrieval:** RAG (Retrieval-Augmented Generation) via ChromaDB  
> **LLM:** Multi-provider with automatic fallback (Gemini → OpenAI → Claude → HuggingFace)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                      │
│  Chatbot.jsx  →  Feedback buttons  →  Detailed feedback modal│
└───────────────────────────┬─────────────────────────────────┘
                            │  HTTP (JSON)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                        │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │  /chat   │  │/feedback │  │/documents│  │ /health  │  │
│   └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘  │
│        │             │                                       │
│   ┌────▼─────────────▼──────────────────────────────────┐  │
│   │              Core Services Layer                     │  │
│   │  ┌─────────────────┐   ┌──────────────────────┐     │  │
│   │  │ ConversationBuf │   │   DocumentRetriever   │     │  │
│   │  │  ferWindowMemory│   │   (ChromaDB RAG)      │     │  │
│   │  └─────────────────┘   └──────────────────────┘     │  │
│   │  ┌──────────────────────────────────────────────┐   │  │
│   │  │         LLM Factory (Multi-provider)          │   │  │
│   │  │  Gemini · OpenAI · Claude · HuggingFace       │   │  │
│   │  └──────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Storage Layer                          │  │
│   │   SQLite (feedback.db)    ChromaDB (vector store)  │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Request Lifecycle — Step by Step

Every user message goes through this exact pipeline:

```
User types a message
        │
        ▼
[1] Frontend (Chatbot.jsx)
    → POST /api/chat/  { message, session_id, user_id }
        │
        ▼
[2] FastAPI Route Handler (routers/chat.py)
    → Validate request (Pydantic: max 4000 chars, strip whitespace)
    → Generate session_id if not provided
    → Generate unique message_id
        │
        ▼
[3] Load Conversation Memory
    → get_or_create_memory(session_id)
    → ConversationBufferWindowMemory(k=10)
    → Returns last 10 Q&A exchanges (in-memory dict)
        │
        ▼
[4] RAG Retrieval (ChromaDB)
    → DocumentRetriever.retrieve_context(user_message)
    → Embeds question → similarity search → top-k chunks
    → Returns: { context: str, sources: [{ title, url, content }] }
        │
        ▼
[5] Build LLM Message Array
    → Past history messages (from memory)
    → Current message (injected with KB context):
       "Context from knowledge base: {kb_context}
        User question: {message}
        Please answer based on the context above."
        │
        ▼
[6] Token Budget Check
    → Count tokens (approx: chars ÷ 4)
    → If total > max_tokens (4000): trim oldest Q&A pairs
    → If > 80% of limit: log a warning
        │
        ▼
[7] LLM Generation (Factory with Fallback)
    → Try preferred provider (or default: Gemini)
    → On failure → try next in fallback_order
    → Returns: { content, provider, tokens_used, finish_reason }
        │
        ▼
[8] Save to Memory
    → memory.save_context({ input: message }, { output: response })
    → Keeps last k=10 exchanges in RAM
        │
        ▼
[9] Persist to Database (SQLite)
    → FeedbackInteraction row:
       user_id, session_id, message_id, question, response,
       provider_used, tokens_used, timestamp
        │
        ▼
[10] Return Response to Frontend
    → { response, sources, session_id, message_id,
        provider_used, token_info }
        │
        ▼
[11] Frontend renders bot message
     → Shows 👍 👎 ✏️ feedback buttons on bot message
```

---

## 3. Conversation Memory Design

### What it is
`ConversationBufferWindowMemory` is a **LangChain in-memory store** that keeps a sliding window of the last **k** question-answer pairs.

### How it works

```
session_memories = {}   ← Python dict, lives in RAM per process

  "session_abc123" → ConversationBufferWindowMemory(k=10)
                          └── InMemoryChatMessageHistory
                               ├── HumanMessage("What is VoxelBox?")
                               ├── AIMessage("VoxelBox is ...")
                               ├── HumanMessage("How do I upload?")
                               ├── AIMessage("Go to ...")
                               └── ...up to last 10 exchanges (20 messages)
```

### The k=10 sliding window rule

| Exchange | Stored? |
|---|---|
| 1st Q&A | ✅ (until pushed out) |
| ... | ✅ |
| 10th Q&A | ✅ |
| 11th Q&A | ✅ — 1st Q&A is **dropped** |

The memory **never expires by time** — it expires by count. Each new exchange pushes out the oldest one once the window is full.

### Why this replaced the old 5-minute window

| | Old (time-based) | Current (buffer-window) |
|---|---|---|
| **Logic** | Context expired after 5 min | Context kept for last 10 exchanges |
| **Dependencies** | Redis + Celery + ChatSession DB | Pure in-memory Python dict |
| **Failure points** | Redis down = lost context | None — no external deps |
| **User experience** | Context lost mid-conversation | Context persists entire session |
| **Infrastructure** | Complex | Zero extra infrastructure |

### Token budget management

Before calling the LLM, the code estimates token usage:

```
approx_tokens = total_characters ÷ 4

if tokens > 4000 (max_tokens):
    → trim pairs from the oldest end until under limit

if tokens > 3200 (80% threshold):
    → log a warning (no action, just monitoring)
```

---

## 4. RAG Pipeline (Retrieval-Augmented Generation)

RAG is what makes the chatbot answer questions about BrainSightAI's products instead of hallucinating.

```
startup:
  DocumentLoader.load_directory("./data")
       ↓
  Split into chunks (size=1000, overlap=200)
       ↓
  HuggingFace Embeddings (all-MiniLM-L6-v2)
       ↓
  Stored in ChromaDB → "BrainSightAIs_knowledge" collection

at query time:
  user_question → embed → cosine similarity → top 4 chunks
       ↓
  chunks injected into LLM prompt as "Context from knowledge base"
```

**Key config (`config.yaml`):**
```yaml
vector_db:
  search:
    k: 4                    # return top 4 chunks
    score_threshold: 0.5    # minimum similarity score
  chunking:
    chunk_size: 1000
    chunk_overlap: 200
```

If retrieval fails (ChromaDB error), the system falls back gracefully:
```python
kb_context = { "context": "No additional context available.", "sources": [] }
```
The conversation still continues — just without KB context.

---

## 5. LLM Multi-Provider Factory

The system supports 4 LLM providers with automatic fallback:

```
Default provider: Gemini
       │
       ▼ (if Gemini fails)
     OpenAI
       │
       ▼ (if OpenAI fails)
     Claude
       │
       ▼ (if Claude fails)
  HuggingFace (Llama 3.3 70B)
```

All providers share the same interface via `llm.base.Message` objects, so swapping providers requires zero changes to the chat logic.

**Config per provider:**
```yaml
providers:
  gemini:
    model: "gemini-2.5-flash"
    temperature: 0.4       # focused, not creative
    max_tokens: 300        # concise answers
    top_p: 0.9
```

---

## 6. Feedback System

### Flow

```
Bot sends a message
    │
    ▼
Frontend shows 3 feedback buttons:
  👍 Thumbs Up     → POST /api/feedback/submit { feedback_type: "thumbs_up" }
  👎 Thumbs Down   → POST /api/feedback/submit { feedback_type: "thumbs_down" }
  ✏️  Write feedback → Opens modal → POST /api/feedback/submit { feedback_type: "detailed_feedback", feedback_text: "..." }
    │
    ▼
Backend updates FeedbackInteraction row:
  feedback_type      = "thumbs_up" | "thumbs_down"
  feedback_comment   = optional text
  feedback_timestamp = datetime.utcnow()
```

### Why feedback is linked to `message_id`

Every chat response returns a unique `message_id` (e.g. `msg_a3f8c1d4b2e7`). The frontend stores this per message. When feedback is submitted, the backend uses `message_id` to find the exact row in `feedback_interactions` and update it — no ambiguity across sessions or users.

---

## 7. Database Schema

Single table: `feedback_interactions` (SQLite → `data/database/feedback.db`)

```
┌──────────────────────────────────────────────────────────────┐
│                    feedback_interactions                      │
├──────────────┬──────────────────────────────────────────────┤
│ id           │ UUID (PK)                                     │
│ user_id      │ String, indexed                               │
│ session_id   │ String, indexed                               │
│ message_id   │ String, UNIQUE, indexed                       │
│ timestamp    │ DateTime, indexed                             │
├──────────────┼──────────────────────────────────────────────┤
│ question     │ Text (user's message)                         │
│ response     │ Text (bot's answer)                           │
│ provider_used│ String (which LLM answered)                   │
│ tokens_used  │ Integer                                       │
├──────────────┼──────────────────────────────────────────────┤
│ feedback_type│ String ("thumbs_up" | "thumbs_down")          │
│ feedback_comment│ Text (optional written feedback)           │
│ feedback_timestamp│ DateTime                                 │
└──────────────┴──────────────────────────────────────────────┘
```

---

## 8. API Endpoints Reference

### Chat
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat/` | Send a message, get AI response |
| `GET` | `/api/chat/history/{session_id}` | Fetch full session history from DB |
| `DELETE` | `/api/chat/history/{session_id}` | Delete session from DB |
| `POST` | `/api/chat/clear/{session_id}` | Clear in-memory context only |
| `GET` | `/api/chat/memory/info/{session_id}` | Inspect memory window state |

### Feedback
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/feedback/submit` | Submit thumbs up/down or written feedback |

### System
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Simple health check |
| `GET` | `/api/health/` | Detailed health (DB, vector store, LLM) |
| `GET` | `/docs` | Swagger UI |

---

## 9. Middleware & Security

Applied in this order (FastAPI reverses `add_middleware` order):

```
Incoming Request
      │
      ▼
[1] CORS Middleware          → Allows listed origins only
      │
      ▼
[2] Malicious Request Block  → Blocks path traversal, .env, wp-admin etc.
      │
      ▼
[3] GZip Compression         → Compresses responses > 1000 bytes
      │
      ▼
[4] Rate Limiter (slowapi)   → 60 req/min, 1000 req/hr per IP
      │
      ▼
[5] Request Logger           → Logs method + path + response status
      │
      ▼
Route Handler
```

---

## 10. Configuration Reference (`config.yaml`)

```
config.yaml
├── app              → name, version, environment, debug flag
├── llm              → default_provider, fallback_order, per-provider settings
├── conversation_memory
│   ├── buffer_window  → k=10 (window size), max_tokens=4000
│   └── token_counting → enabled=true, warning_threshold=0.8
├── embeddings       → HuggingFace all-MiniLM-L6-v2
├── vector_db        → ChromaDB, chunk_size=1000, top_k=4
├── documents        → supported formats, data_dir, upload_dir
├── api              → host, port, CORS origins, rate limits
├── logging          → level, file rotation, console
├── system_prompt    → Full persona/behaviour instructions for the LLM
└── monitoring       → usage/cost/performance tracking flags
```

---

## 11. Startup Sequence

When the backend starts (`uvicorn main:app`):

```
1. Load config.yaml → validate all config models (Pydantic)
2. init_db()        → create SQLite tables if they don't exist
3. DocumentRetriever() → connect to ChromaDB
   ├── If 0 docs → load from ./data/ → chunk → embed → store
   └── If docs exist → ready (log count)
4. LLM Factory      → scan which providers have API keys
5. Log available providers + default provider
6. Start serving requests
```

---

## 12. Frontend → Backend Contract

**Chat Request:**
```json
POST /api/chat/
{
  "message": "How do I upload a scan?",
  "session_id": "session_1747550000123",
  "user_id": "user_1747550000456"
}
```

**Chat Response:**
```json
{
  "response": "To upload a scan in VoxelBox...",
  "sources": [{ "title": "Upload Guide", "url": "...", "content": "..." }],
  "session_id": "session_1747550000123",
  "message_id": "msg_a3f8c1d4b2e7",
  "success": true,
  "provider_used": "gemini",
  "tokens_used": 142,
  "token_info": {
    "message_tokens": 12,
    "history_tokens": 45,
    "total_tokens": 199,
    "max_tokens": 4000,
    "percentage": 4.975,
    "warning": false
  }
}
```

**Feedback Request:**
```json
POST /api/feedback/submit
{
  "message_id": "msg_a3f8c1d4b2e7",
  "feedback_type": "thumbs_up",
  "feedback_comment": null
}
```
