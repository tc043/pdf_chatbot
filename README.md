# Enterprise PDF RAG Chatbot

An enterprise-grade, multi-tenant PDF retrieval (RAG) and conversation system. It transitions the application from a single-process prototype to a modern, decoupled microservices architecture designed to scale.

---

## Architecture Overview

```
pdf_chatbot/
├── backend/
│   ├── app/
│   │   ├── core/           # Database, Security & Config
│   │   ├── models.py       # SQLAlchemy Models (pgvector)
│   │   ├── schemas.py      # Pydantic Schemas
│   │   ├── services/       # Embeddings, storage & LLM APIs
│   │   ├── routers/        # Auth, documents, chats API routes
│   │   └── worker.py       # Celery async worker logic
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/            # Next.js App Router UI Pages
│   │   ├── components/     # PDF viewer & chat interfaces
│   │   └── lib/            # Axios API client & SSE parser
│   ├── Dockerfile
│   ├── package.json
│   └── tailwind.config.js
├── docker-compose.yml       # Service Orchestrator
├── nginx.conf              # Reverse proxy & Rate limiter
└── README.md
```

### Key Capabilities
*   **Decoupled Frontend:** Built with Next.js (App Router, Tailwind CSS, TypeScript, Lucide Icons) for a smooth document workspace experience.
*   **Decoupled Backend:** Built with FastAPI supporting async requests and Server-Sent Events (SSE) streaming.
*   **Relational + Vector Storage:** Uses PostgreSQL backed by the `pgvector` extension, merging users, chat history, and document embeddings in a single database.
*   **Async Processing Worker:** Heavy indexing/embeddings computations are backgrounded using Celery and Redis to prevent API thread blocking.
*   **Observability:** Built-in Prometheus instrumentation and structured JSON logs tracking Request IDs across all services.
*   **Secure Gateway:** NGINX reverse proxy providing path-based routing and protection against API spam (rate limits).

---

## Setup & Running the Application

### Prerequisites
*   **Docker** (v20+)
*   **Docker Compose** (v2+)
*   **OpenAI API Key** (optional, fallback to local CPU-based embedding/LLM operations is supported automatically)

### 1. Configure the Environment
Create a `.env` file in the root directory (or inject variables directly into your host terminal env):

```env
OPENAI_API_KEY=sk-your-openai-api-key
```

### 2. Launch the Services
Start the entire stack (Postgres, Redis, MinIO, Backend, Worker, Frontend, NGINX) using Docker Compose:

```bash
docker compose up --build -d
```

### 3. Open the Dashboard
Once the containers are built and initialized, navigate to:
*   **Workspace Webpage:** `http://localhost` (Routes through NGINX)
*   **FastAPI API Docs:** `http://localhost/api/v1/docs` (Interactive Swagger interface)
*   **MinIO Console:** `http://localhost:9001` (Inspect PDF uploads inside the S3 storage bucket)
*   **Backend Metrics:** `http://localhost/api/v1/metrics` (Prometheus raw scraper page)

### 4. Running Local & Native Host Setup (Without Docker)

You can run the entire microservices stack bare-metal (natively on your host machine) using a local SQLite database fallback (which implements in-memory cosine similarity vector calculation, keeping database operations zero-dependency):

#### Prerequisites
*   Python 3.9+ & Node.js v18+
*   Redis server and MinIO binary installed/runnable on the host.

#### Quick Start (All-in-One Local Script)
We provide an automated startup script that starts local Redis and MinIO storage, sets up database tables inside SQLite, spawns the Celery async worker, launches the FastAPI backend, and hosts the Next.js frontend:
```bash
# Make the startup script executable
chmod +x start_services_local.sh

# Boot the stack natively in the background
./start_services_local.sh
```

Once running, monitor backend logs with `tail -f backend.log` and Celery tasks with `tail -f worker.log`.

#### Run Mock Unit Tests
```bash
python3 backend/test_api_mock.py
```

#### Run Live End-to-End RAG Integration Tests
With the native services running, execute the live integration test to verify the complete user registration -> PDF creation/upload -> Celery indexing -> RAG search -> SSE token streaming pipeline:
```bash
python3 test_live_rag.py
```

---

## User Flow Walkthrough

1.  **Register / Log In:** Open the webpage and sign up for a new account. The system hashes credentials and returns a secure JWT bearer token.
2.  **Upload PDF:** Drag-and-drop or select a document in the PDF panel. 
    *   The backend saves the PDF blob into S3 (MinIO) and returns a `202 Accepted` status.
    *   A Celery task splits the document, calculates embeddings (OpenAI text-embedding-3 or local SentenceTransformers), pads them to 1536 dimensions, and inserts the chunks into PostgreSQL.
3.  **Start Chatting:** The right sidebar logs conversations. Type a query.
    *   FastAPI intercepts the query, computes the question vector, and performs a similarity join via pgvector.
    *   A Server-Sent Events (SSE) stream emits retrieved document citations first (auto-scrolling/navigating the left PDF viewer to the relevant page).
    *   The LLM stream is printed in real-time.
