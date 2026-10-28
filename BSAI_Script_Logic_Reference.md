# BSAI Chatbot — Per-Script Logic Reference

> Detailed explanation of every script file in the project.
> For the high-level architecture overview, see `BSAI_Chatbot_Architecture.md`.

---

## 📄 `backend/main.py` — Application Entry Point

**Purpose:** Bootstraps the entire FastAPI application. Wires together all middleware, routers, and startup logic.

**Key logic:**

```
lifespan() context manager (runs at startup)
  ├── init_db()             → create SQLite tables if missing
  ├── DocumentRetriever()   → connect to ChromaDB
  │     └── if 0 docs → auto-load from ./data/ directory
  └── LLM Factory scan      → log which providers have API keys

Middleware stack (added in reverse, executed top-to-bottom):
  [1] CORS             → allow listed origins
  [2] Malicious path   → block traversal, .env, wp-admin etc.
  [3] GZip             → compress responses > 1KB
  [4] Rate limiter     → 60/min per IP via slowapi
  [5] Request logger   → log every method + path + status code

Router registration:
  /api/chat/*      → chat_router
  /api/feedback/*  → feedback_router
  /api/documents/* → documents_router
  /api/health/*    → health_router
  /api/config/*    → config_router
```

**Design decisions:**
- Uses `@asynccontextmanager lifespan` instead of deprecated `@app.on_event("startup")` — guarantees clean startup AND shutdown.
- **Windows fix:** Forces `stdout`/`stderr` to UTF-8 encoding at startup to prevent emoji/unicode crashes on Windows terminals.
- Middleware is registered in **reverse order** (FastAPI quirk) — CORS is added last so it executes first.

---

## 📄 `backend/routers/chat.py` — Core Chat Logic

**Purpose:** The brain of the chatbot. Handles every user message end-to-end.

**Internal classes:**

```python
ConversationBufferWindowMemory(k=10)
  # Drop-in wrapper around LangChain's InMemoryChatMessageHistory
  # save_context()          → adds Q+A pair to history
  # load_memory_variables() → returns last k*2 messages (k pairs)
  # clear()                → wipes history for session

session_memories: Dict[str, ConversationBufferWindowMemory]
  # Global in-process dict keyed by session_id
  # Lives in RAM — no disk, no Redis, no DB reads for context
```

**Token management logic:**
```
count_tokens_in_messages(messages)
  → total_chars ÷ 4   (approximation: 4 chars ≈ 1 token)

check_token_limit(messages, max_tokens=4000, threshold=0.8)
  → returns { token_count, percentage, warning, exceeds_limit }

If exceeds_limit:
  → loop: remove oldest Q&A pair (2 messages at a time)
  → repeat until under limit
```

**Pydantic request validation:**
```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    # @validator sanitizes whitespace before processing
```

**Endpoints:**

| Endpoint | Logic |
|---|---|
| `POST /` | Full 11-step chat pipeline |
| `GET /history/{id}` | Query `FeedbackInteraction` table ordered by timestamp |
| `DELETE /history/{id}` | Delete from memory dict + DB rows |
| `POST /clear/{id}` | Delete from memory dict only (DB history kept) |
| `GET /memory/info/{id}` | Inspect k, token count, message list for debugging |

---

## 📄 `backend/routers/feedback.py` — Feedback Collection

**Purpose:** Accepts user feedback on bot responses and stores it in the DB.

**Logic flow:**
```
POST /api/feedback/submit
  ├── Validate: feedback_type must be "thumbs_up" or "thumbs_down"
  ├── Lookup: FeedbackInteraction WHERE message_id = ?
  │     └── 404 if message not found
  └── UPDATE row:
        feedback_type      = request value
        feedback_comment   = optional text
        feedback_timestamp = now()
```

**Why `message_id` not `session_id`?** Each bot reply has a unique `message_id`. Linking feedback to `message_id` pinpoints the exact response that was rated — no ambiguity across sessions or users.

**Design:** Feedback is an UPDATE on an existing row, not a new INSERT. This keeps one clean row per exchange covering both the conversation content and the feedback rating.

---

## 📄 `backend/routers/health.py` — Health Checks

**Purpose:** Provides 3 endpoints for monitoring and deployment readiness.

| Endpoint | What it checks | Use case |
|---|---|---|
| `GET /api/health/` | LLM providers + ChromaDB + doc count | Full monitoring dashboards |
| `GET /api/health/ready` | At least 1 LLM provider available | Kubernetes `readinessProbe` |
| `GET /api/health/live` | Returns `{alive: true}` always | Kubernetes `livenessProbe` |

**Design:** The full `/` check catches partial failures (e.g. ChromaDB down but LLMs working) — it still returns `status: healthy` for the API itself with degraded sub-service status in the body. This prevents false positives in uptime monitors.

---

## 📄 `backend/routers/documents.py` — Document Management

**Purpose:** Allows uploading new documents to the knowledge base at runtime without restarting the server.

**Flow:**
```
POST /api/documents/upload
  → validate file (type + size via DocumentLoader.validate_file)
  → save to ./uploads/
  → DocumentLoader.load_document()
  → VectorStore.add_documents() → chunk → embed → store in ChromaDB
  → return { chunks_added, filename }

DELETE /api/documents/
  → VectorStore.delete_collection()
  → wipes entire ChromaDB collection
```

---

## 📄 `backend/routers/config.py` — Config Inspection

**Purpose:** Read-only API to inspect the running configuration without SSH access.

Returns sanitized config — **API keys are never exposed**, only the env var name (e.g. `"GOOGLE_API_KEY"` not the actual key value).

---

## LLM Layer (`backend/llm/`)

---

## 📄 `backend/llm/base.py` — Provider Contract (Abstract Base Class)

**Purpose:** Defines the interface every LLM provider must implement. Enforces a consistent API across Gemini, OpenAI, Claude, and HuggingFace.

**Key types:**
```python
@dataclass
class Message:
    role: str       # "user" | "assistant" | "system"
    content: str
    def to_dict() → { role, content }

@dataclass
class LLMResponse:
    content: str          # the actual answer text
    model: str            # e.g. "gemini-2.5-flash"
    provider: str         # e.g. "gemini"
    tokens_used: int      # reported by the API
    finish_reason: str    # "stop" | "error" | "length"
    error: str            # error message if failed

class BaseLLMProvider(ABC):
    @abstractmethod validate_config()      → init API client, raise on failure
    @abstractmethod generate_response()    → sync call → LLMResponse
    @abstractmethod stream_response()      → async generator of text chunks
    format_messages()                      → prepend system prompt, format list
    _handle_error()                        → wrap exceptions into LLMResponse(finish_reason="error")
```

**Design pattern:** By enforcing `ABC + abstractmethod`, adding a new LLM provider only requires:
1. Subclassing `BaseLLMProvider`
2. Implementing 3 methods
3. Registering it in `factory.py`'s `PROVIDERS` dict

No other file needs to change.

---

## 📄 `backend/llm/factory.py` — Provider Orchestrator

**Purpose:** Creates provider instances at startup, manages them as singletons, and implements the fallback strategy.

**Startup initialization:**
```
_initialize_providers()
  for each provider in config.yaml:
    if enabled = false → skip
    if no API key in environment → skip (log warning)
    else → instantiate provider class → add to self._providers dict
```

**Fallback strategy (`generate_with_fallback`):**
```
Build ordered list to try:
  1. preferred_provider  (if passed in request AND available)
  2. default_provider    (config: gemini)
  3. fallback_order list (openai → gemini → huggingface → claude)
  (dedup: skip already-added providers)

Try each in order:
  → call provider.generate_response()
  → if finish_reason != "error" → return immediately ✅
  → if error → log warning + try next provider

If ALL fail → return LLMResponse(finish_reason="error", content="sorry...")
```

**Singleton:** `_factory` global ensures providers are initialized once per process — not once per request. Provider SDK clients are expensive to create.

---

## 📄 `backend/llm/gemini_provider.py` / `openai_provider.py` / `claude_provider.py` / `huggingface_provider.py`

**Purpose:** Concrete implementations of `BaseLLMProvider` for each LLM service.

Each file follows the same 3-method pattern:
```python
class GeminiProvider(BaseLLMProvider):

    def validate_config(self):
        # initialize SDK client with self.api_key
        # raise ValueError if setup fails → excluded from factory

    def generate_response(self, messages, system_prompt):
        try:
            formatted = self.format_messages(messages, system_prompt)
            raw = self.client.call(formatted, temperature=..., max_tokens=...)
            return LLMResponse(
                content      = raw.text,
                provider     = "gemini",
                tokens_used  = raw.usage.total_tokens,
                finish_reason= "stop"
            )
        except Exception as e:
            return self._handle_error(e, "generate")

    async def stream_response(self, messages, system_prompt):
        # async generator yielding text chunks for streaming UIs
```

**HuggingFace difference:** Uses the HuggingFace Inference API via HTTP REST (not a native SDK). `stream: false` in config because the endpoint returns the full response synchronously.

---

## Vector DB Layer (`backend/vector_db/`)

---

## 📄 `backend/vector_db/embeddings.py` — Embedding Model Manager

**Purpose:** Returns a singleton embedding model used both when indexing documents AND when searching. Using the same model for both operations is mandatory for correct similarity matching.

**Logic:**
```
get_embeddings()  ← singleton, only created once per process
  if config.embeddings.provider == "openai":
      → OpenAIEmbeddings(model="text-embedding-3-small")  [1536 dims]
  elif "huggingface":
      → HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")  [384 dims]

  on any failure:
      → fallback to HuggingFace all-MiniLM-L6-v2 (runs locally, no API key)
```

**Why `all-MiniLM-L6-v2`?**
- Runs entirely locally — no API calls, no latency, no cost
- 384-dimension vectors = fast similarity search
- Strong semantic similarity for technical documentation
- Good enough accuracy for a support chatbot's knowledge base

---

## 📄 `backend/vector_db/store.py` — ChromaDB Wrapper

**Purpose:** Abstracts all ChromaDB operations behind a clean Python API.

**Methods:**

```python
add_documents(documents: List[Document]) → int
  # 1. RecursiveCharacterTextSplitter.split_documents()
  #    → chunks (size=1000 chars, overlap=200 chars)
  # 2. vectordb.add_documents(chunks)
  #    → embed each chunk using embeddings model
  #    → store vector + text in ChromaDB collection
  # returns: number of chunks stored

search(query, k=4) → List[Document]
  # embed query string
  # cosine similarity search against stored vectors
  # return top-k most similar document chunks

search_with_score(query, k, threshold=0.5) → List[(Document, float)]
  # same as search but returns similarity score per result
  # filters out docs below score_threshold
  # available for future "confidence" display features

delete_collection() → None
  # wipes entire ChromaDB collection
  # called by DELETE /api/documents/

get_document_count() → int
  # used at startup to decide whether to auto-load docs
  # if 0 → trigger DocumentLoader.load_directory()
```

**Why `chunk_overlap=200`?** Overlapping chunks ensure sentences that span a chunk boundary aren't semantically cut in half. Both adjacent chunks contain the boundary content, preserving meaning at the edges.

---

## 📄 `backend/vector_db/retriever.py` — RAG Context Builder

**Purpose:** Thin orchestration layer that calls `VectorStore.search()` and formats results into the exact shape the chat router needs.

**Logic:**
```
retrieve_context(query, k=4)
  → vector_store.search(query, k)
  → for each doc result:
       context_parts.append("[1] {page_content}")
       sources.append({ title, url, content[:200]+"..." })
  → return {
       context: "[1] ...\n\n[2] ...",    ← injected into LLM prompt
       sources: [ {title, url, content} ] ← shown in frontend
     }
```

**Why number the context chunks `[1], [2]...`?**
- Helps the LLM anchor its answer to specific retrieved passages
- Reduces hallucination — LLM says "based on [1]..." instead of inventing
- Makes source attribution more natural

---

## Utils (`backend/utils/`)

---

## 📄 `backend/utils/document_loader.py` — Multi-Format File Loader

**Purpose:** Loads documents from disk in any supported format and returns LangChain `Document` objects ready for chunking and embedding.

**Supported formats and their loaders:**

| Extension | LangChain Loader | Notes |
|---|---|---|
| `.pdf` | `PyPDFLoader` | Extracts text page-by-page, preserves structure |
| `.docx` | `Docx2txtLoader` | Extracts plain text from Word documents |
| `.txt` | `TextLoader` | Raw UTF-8 text read |
| `.md` | `UnstructuredMarkdownLoader` | Parses markdown heading structure |

**Methods:**
```python
load_document(file_path) → List[Document] | None
  # loads single file using format-specific loader
  # adds metadata: { source: filename, file_path: full_path }
  # metadata is stored in ChromaDB and returned in search results

load_directory(directory) → List[Document]
  # globs for each supported extension
  # loads all matching files, skips failures gracefully
  # called at startup if ChromaDB collection is empty

validate_file(file_path, max_size) → (bool, error_msg)
  # checks: exists? supported format? under 10MB size limit?
  # called by /api/documents/upload before loading
```

---

## 📄 `backend/utils/logger.py` — Logging Setup

**Purpose:** Creates a configured Python logger with both console and rotating file output, driven entirely by `config.yaml`.

**Logic:**
```
setup_logger(name)
  → read config.logging (level, format, file path, max_bytes, backup_count)
  → clear any existing handlers  ← prevents duplicate logs on uvicorn hot-reload
  → add StreamHandler(stdout)          if config.console.enabled = true
  → add RotatingFileHandler(./logs/)   if config.file.enabled = true
       └── maxBytes=10MB, backupCount=5
  → return configured logger
```

**RotatingFileHandler behaviour:**
When `app.log` reaches 10MB:
- `app.log` → `app.log.1`
- `app.log.1` → `app.log.2`
- ... up to `app.log.5` (oldest deleted)
- New `app.log` started fresh

This prevents disk exhaustion in long-running production deployments.

---

## Database Layer (`backend/database/`)

---

## 📄 `backend/database/models.py` — Data Model

**Purpose:** Defines the single SQLAlchemy ORM table used by the system.

```python
class FeedbackInteraction(Base):
    __tablename__ = 'feedback_interactions'

    # Identity (all indexed for fast querying)
    id          → UUID primary key (auto-generated on insert)
    user_id     → who sent the message
    session_id  → which conversation session (groups messages)
    message_id  → unique per bot reply (used for feedback linking)
    timestamp   → when the exchange happened

    # Chat content
    question    → what the user asked (Text, no length limit)
    response    → what the bot answered (Text, no length limit)
    provider_used → which LLM answered (gemini/openai/claude/huggingface)
    tokens_used   → approximate token count for cost tracking

    # Feedback (populated later by user action)
    feedback_type      → "thumbs_up" | "thumbs_down" (nullable)
    feedback_comment   → optional written feedback text (nullable)
    feedback_timestamp → when feedback was submitted (nullable)
```

**Design decision — one table:** Chat history AND feedback live in the same row. Feedback is just an UPDATE. No JOINs needed, simple for SQLite, easy to query for analytics.

---

## 📄 `backend/database/connection.py` — DB Connection Manager

**Purpose:** Creates the SQLAlchemy engine + session factory and exposes the FastAPI dependency injector.

```python
engine = create_engine(
    "sqlite:///data/database/feedback.db",
    connect_args={"check_same_thread": False}  # required for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(engine)  # creates table if not exist (idempotent)

def get_db():  # used as FastAPI Depends() injection
    db = SessionLocal()
    try:
        yield db       # give session to route handler
    finally:
        db.close()     # always close even on exception
```

**`check_same_thread=False`:** SQLite restricts connections to the thread that created them. FastAPI's async workers operate across threads, so this flag is mandatory to prevent `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`.

---

## 📄 `backend/config.py` — Config System

**Purpose:** Loads `config.yaml` once at startup, validates every field with Pydantic, and exposes typed accessor functions used across the codebase.

**Why Pydantic validation?**
If `config.yaml` has a wrong type (e.g. `temperature: "hot"` instead of `0.4`), the app crashes immediately at startup with a clear error — not silently mid-request with a cryptic `TypeError`.

**Singleton pattern:**
```python
_config: Optional[Config] = None

def get_config() → Config:
    if _config is None:
        _config = load_config()   # read YAML + validate once
    return _config                # return cached object on all subsequent calls
```

**Config model hierarchy:**
```
Config
 ├── AppConfig         (name, version, environment, debug)
 ├── LLMConfig         (default_provider, fallback_order, providers dict)
 │     └── ProviderConfig × 4  (model, api_key_env, temperature, max_tokens...)
 ├── ConversationMemoryConfig
 │     ├── BufferWindowConfig   (k, max_tokens, memory_key)
 │     └── TokenCountingConfig  (enabled, encoding, warning_threshold)
 ├── EmbeddingsConfig  (provider, providers dict)
 ├── VectorDBConfig    (type, persist_directory, search, chunking)
 ├── DocumentsConfig   (supported_formats, max_file_size, data_dir)
 ├── APIConfig         (host, port, cors, rate_limit, timeout)
 ├── LoggingConfig     (level, format, file, console)
 ├── MonitoringConfig  (enabled, track_usage, track_costs, alerts)
 └── system_prompt     (str — full persona instructions for the LLM)
```

**Convenience accessors** (used across the codebase to avoid re-loading):
```python
get_api_config()                 → config.api
get_llm_config()                 → config.llm
get_conversation_memory_config() → config.conversation_memory
get_provider_api_key(name)       → os.getenv(provider.api_key_env)
get_embedding_api_key()          → os.getenv(embeddings.provider.api_key_env)
```

---

## 📄 `backend/tasks.py` — Daily Report Utility

**Purpose:** Standalone DB reporting function. No Celery, no Redis, no external dependencies.

```python
generate_daily_report(date=None) → dict
  # queries feedback_interactions for yesterday's date (or given date)
  # returns:
  {
    "date": "2026-05-17",
    "total_messages": 142,
    "unique_users": 31,
    "unique_sessions": 38,
    "total_tokens": 28400,
    "provider_stats": { "gemini": 130, "openai": 12 },
    "avg_messages_per_user": 4.58,
    "avg_messages_per_session": 3.74
  }
```

**How to use:**
```bash
# Run manually
python tasks.py

# Wire into any scheduler (cron, APScheduler, etc.)
from tasks import generate_daily_report
report = generate_daily_report()
```

---

## Frontend (`frontend/src/`)

---

## 📄 `frontend/src/components/Chatbot.jsx` — React Chat UI

**Purpose:** The complete chatbot widget — toggle button, chat window, message list, input field, typing indicator, and feedback system.

**State variables:**
```javascript
isOpen        → bool: show/hide the chat window
messages      → array: { id, text, sender, timestamp, messageId, sources, feedback }
input         → string: current typed text in input box
isTyping      → bool: show animated typing dots while waiting for API
sessionId     → string: generated once on mount (session_${Date.now()})
feedbackModal → string|null: messageId of message with open feedback modal
feedbackText  → string: text being typed in feedback modal textarea
```

**Core functions:**

```javascript
sendMessage()
  → optimistic UI: append user message to state immediately
  → POST /api/chat/ { message, session_id, user_id }
  → on success: append bot message with messageId, sources
  → on network error: append error message bubble
  → finally: setIsTyping(false)

handleFeedback(messageId, type)
  → POST /api/feedback/submit { message_id, feedback_type }
  → on success: update message in state → replace buttons with emoji

handleDetailedFeedback()
  → POST /api/feedback/submit { message_id, "detailed_feedback", feedback_text }
  → close modal, mark message with ✏️

clearChat()
  → reset messages to just the welcome message
  → generate new session_id (new session = fresh in-memory context)
  → note: does NOT call /api/chat/clear/{id} — old memory is just abandoned
```

**API URL auto-detection:**
```javascript
const API_BASE_URL =
  process.env.REACT_APP_API_URL           // production: set in .env
  || `http://${window.location.hostname}:8000`  // dev/staging: auto-detect
```
The same frontend build works on `localhost`, LAN IP `172.16.x.x`, or any production domain without code changes.

**Typing indicator:** Three animated `<span>` dots in `.typing-indicator` — purely CSS `@keyframes` animation, no JavaScript timer involved.

**Feedback button visibility logic:**
```jsx
{message.sender === 'bot'
  && message.messageId          // only bot messages with a messageId
  && !message.feedback          // hide once feedback is submitted
  && (
    <FeedbackButtons />
  )
}
```

---

## 📄 `frontend/src/components/Chatbot.css` — UI Styles

Contains all styles for:
- `.chatbot-toggle` — floating action button (bottom-right)
- `.chatbot-container` — main chat window (positioned above toggle)
- `.chatbot-header` — top bar with avatar, clear, close buttons
- `.chatbot-messages` — scrollable message area
- `.message.user` / `.message.bot` — message bubbles with distinct styling
- `.typing-indicator` — three-dot animation via CSS `@keyframes`
- `.feedback-buttons` — thumbs up/down/edit row under bot messages
- `.feedback-modal-overlay` / `.feedback-modal` — detailed feedback modal

---

## File Map Summary

```
backend/
├── main.py                     ← App bootstrap, middleware, router wiring
├── config.py                   ← YAML config loading + Pydantic validation
├── config.yaml                 ← All configuration values
├── tasks.py                    ← Daily report utility function
│
├── routers/
│   ├── chat.py                 ← Chat endpoint + memory management (CORE)
│   ├── feedback.py             ← Feedback submit endpoint
│   ├── health.py               ← Health/ready/live check endpoints
│   ├── documents.py            ← Upload/delete knowledge base docs
│   └── config.py               ← Read-only config inspection endpoint
│
├── llm/
│   ├── base.py                 ← Abstract provider interface (Message, LLMResponse)
│   ├── factory.py              ← Provider manager + fallback strategy (CORE)
│   ├── gemini_provider.py      ← Google Gemini implementation
│   ├── openai_provider.py      ← OpenAI GPT implementation
│   ├── claude_provider.py      ← Anthropic Claude implementation
│   └── huggingface_provider.py ← HuggingFace Inference API implementation
│
├── vector_db/
│   ├── embeddings.py           ← Embedding model singleton (HuggingFace/OpenAI)
│   ├── store.py                ← ChromaDB wrapper (add, search, delete)
│   └── retriever.py            ← RAG context builder (CORE)
│
├── utils/
│   ├── document_loader.py      ← Multi-format file loader (PDF/DOCX/TXT/MD)
│   └── logger.py               ← Rotating file + console logger setup
│
└── database/
    ├── models.py               ← FeedbackInteraction SQLAlchemy model
    └── connection.py           ← Engine, session factory, FastAPI dependency

frontend/src/components/
├── Chatbot.jsx                 ← Complete chat widget (CORE)
└── Chatbot.css                 ← All widget styles
```
