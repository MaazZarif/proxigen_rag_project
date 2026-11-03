# Proxigen RAG Support Chatbot

A chatbot for Proxigen's website. It answers questions about Proxigen and its products using **only the company's own text files** (`data/cleaned/`). If the answer isn't there, it says so and points to info@proxigen.net, so it doesn't make things up.

## Contents

1. [How it works](#how-it-works)
2. [What each node does](#what-each-node-does)
3. [Tools and why](#tools-and-why)
4. [API keys](#api-keys)
5. [What the backend sets up by itself](#what-the-backend-sets-up-by-itself)
6. [Three ways to run it](#three-ways-to-run-it)
   - [Option 1: Docker image (everything in one command)](#option-1-docker-image-everything-in-one-command)
   - [Option 2: Manual, Qdrant and PostgreSQL in Docker](#option-2-manual-qdrant-and-postgresql-in-docker)
   - [Option 3: Manual, PostgreSQL installed on your computer](#option-3-manual-postgresql-installed-on-your-computer)
   - [Shared steps for Options 2 and 3](#shared-steps-for-options-2-and-3)

---

## How it works

1. The user types a question in the chat widget.
2. The widget sends it to the FastAPI backend (`POST /chat`).
3. The backend checks it for safety and rewrites it using the chat history, so follow-up questions work.
4. It searches the knowledge base: meaning search plus keyword search, then re-ranking.
5. An AI model writes the answer from the found passages only. A second check confirms every claim is supported, otherwise the user gets a safe fallback reply.
6. The conversation is saved so follow-ups keep their context.

---

## What each node does

Every question runs through a workflow of 13 small steps called **nodes** (built with LangGraph, defined in `src/proxigen_rag/graph/graph.py`). Each node does one job and passes its result to the next.

| # | Node | What it does, in plain words |
|---|---|---|
| 1 | `input_guardrail` | The security guard. Rejects empty messages, messages over 500 characters, and odd control characters. An AI classifier also blocks attempts to trick the bot (prompt injection). If the check itself fails, the message is blocked to be safe. |
| 2 | `guardrail_fallback` | Gives a fixed reply ("I couldn't process that message...") when the guard blocks a message. |
| 3 | `save_question` | Adds the question to the conversation history and clears the leftovers of the previous question. |
| 4 | `prepare_query` | Rewrites the question so it stands on its own, using recent chat history ("how much is it?" becomes "What is the price of the CTM01?"). If key information is missing, it asks the user a short clarifying question instead. |
| 5 | `query_analyzer` | Decides the route: small talk (greetings, thanks) goes straight to a reply, everything else goes to the knowledge base. When unsure, it uses the knowledge base. |
| 6 | `direct_answer` | Replies to greetings and thanks. It never states Proxigen facts. |
| 7 | `retrieval_node` | Searches Qdrant in two ways at once (by meaning and by keywords), merges the results, and returns the 5 best passages. |
| 8 | `reranker` | A second opinion on relevance. A small local AI model re-scores the passages against the question and keeps the best 5. |
| 9 | `answer_with_evidence` | Writes the answer in plain language **using only the found passages**, and rates the evidence as *sufficient*, *partial* (some parts missing), or *insufficient*. |
| 10 | `retry_query` | If the evidence was insufficient, rewrites the search query once and searches again. |
| 11 | `verify_grounding` | The fact-checker. Checks that every important claim in the answer is supported by the passages. Unsupported answers are thrown away. |
| 12 | `fallback_answer` | Writes a short, friendly reply when there is no verified answer: it asks what the user meant if the question was unclear, or says the information isn't available and suggests info@proxigen.net. |
| 13 | `save_answer` | Saves the final answer to the conversation history. Every path ends here. |

```mermaid
flowchart TD
    START([Question]) --> IG[input_guardrail]
    IG -- allowed --> SQ[save_question]
    IG -- blocked --> GF[guardrail_fallback]
    SQ --> PQ[prepare_query]
    PQ -- needs clarification --> SA[save_answer]
    PQ -- clear --> QA[query_analyzer]
    QA -- small talk --> DA[direct_answer]
    QA -- product or factual --> RN[retrieval_node]
    DA --> SA
    RN --> RR[reranker]
    RR --> AE[answer_with_evidence]
    AE -- sufficient or partial --> VG[verify_grounding]
    AE -- insufficient, first try --> RQ[retry_query]
    AE -- insufficient, after retry --> FB[fallback_answer]
    RQ --> RN
    VG -- grounded --> SA
    VG -- ungrounded --> FB
    FB --> SA
    GF --> SA
    SA --> END([Answer])
```

**In short:** the bot only answers when it found evidence **and** the evidence checks out. Otherwise it falls back to a polite "I don't have that information", so unverified claims never reach the user.

---

## Tools and why

| Tool | Why it's there |
|---|---|
| Python 3.13 and **uv** | The language, plus a tool that installs the exact package versions in one command. |
| **FastAPI** | The web server that exposes `/chat` and `/health`. |
| **LangGraph and LangChain** | Run the answer pipeline as controlled steps (safety, search, answer, verify, fallback). |
| **Groq API** (cloud) | The language models that write and check answers. **The only required API key.** |
| **Nomic embedding model** (local) | Turns text into numbers so the app can search by meaning. It downloads itself on first run, with no key and no extra software. |
| **Qdrant** | The database that stores and searches the knowledge base. |
| **PostgreSQL** | Stores conversation memory. |
| **Docker** | Runs Qdrant and PostgreSQL, or the whole project with one command. |
| Chat widget (HTML, CSS, JavaScript) | The chat window. It needs no build step. |

---

## API keys

- **Required:** a Groq API key, free from https://console.groq.com/keys.
- **Optional:** a LangSmith key, for developer debugging only.
- **Not needed:** OpenAI, Cerebras, Ollama.

Never share or commit your `.env` file. It holds your keys and is already excluded from git.

---

## What the backend sets up by itself

When the backend starts (`main.py`), it creates the Qdrant collection if it is missing, creates the PostgreSQL tables, and builds the answer workflow.

It does **not** load the documents. That is the *ingestion* command: run it once, and again whenever the files in `data/cleaned/` change. If ingestion is skipped, the bot answers everything with "I don't have that information." (Option 1 runs ingestion for you automatically.)

---

## Three ways to run it

| | Option 1: Docker image | Option 2: Manual + Docker databases | Option 3: Manual + local PostgreSQL |
|---|---|---|---|
| **You install** | Git, Docker Desktop | Git, uv, Docker Desktop, VS Code | Git, uv, Docker Desktop, VS Code, PostgreSQL |
| **Python/uv needed?** | No | Yes | Yes |
| **Qdrant runs in** | Docker | Docker | Docker |
| **PostgreSQL runs in** | Docker | Docker | Installed on your computer |
| **Best for** | Easiest, nothing to configure | Developers who want to run the code directly | You already have PostgreSQL |

For all options you need a **Groq API key** (see above). **Qdrant Cloud will not work**, because the code has no setting for a Qdrant API key, so Qdrant always runs locally in Docker.

---

### Option 1: Docker image (everything in one command)

Docker builds the backend image on your computer and starts everything: PostgreSQL, Qdrant, the backend, and the chat page. It also indexes the documents automatically.

**Install first:** [Git](https://git-scm.com/downloads) and [Docker Desktop](https://www.docker.com/products/docker-desktop/) (open it and wait until it says it is running).

```powershell
git clone https://github.com/MaazZarif/proxigen_rag_project.git
cd proxigen_rag_project
copy .env.example .env          # Linux/macOS: cp .env.example .env
notepad .env                    # Linux/macOS: nano .env
```

In `.env`, set `GROQ_API_KEY` (your real key) and `POSTGRES_PASSWORD` (any password, letters and numbers only). Leave everything else as it is. Then:

```powershell
docker compose up --build
```

The **first run takes several minutes** (it builds the image and downloads about 0.7 GB of AI models). When the log shows `Proxigen RAG API started successfully`, open **http://127.0.0.1:5500** and click **Chat with us**.

| Task | Command |
|---|---|
| Stop (keeps your data) | `Ctrl+C`, then `docker compose down` |
| Start again | `docker compose up` |
| Run in the background | `docker compose up -d` |
| View backend logs | `docker compose logs -f backend` |
| Re-index after editing `data/cleaned/` | `docker compose up --build` |
| Delete everything, including saved data and downloaded models | `docker compose down -v` |

Do not use Live Server with this option, because both want port 5500.

---

### Option 2: Manual, Qdrant and PostgreSQL in Docker

You run the Python code yourself; only the two databases run in Docker.

**Install first:**
1. **Git**
2. **uv**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`). Reopen the terminal afterwards.
3. **Docker Desktop** (open and running)
4. **VS Code** with the **Live Server** extension
5. A **Groq API key**

**Step 1. Clone the project**
```powershell
git clone https://github.com/MaazZarif/proxigen_rag_project.git
cd proxigen_rag_project
```

**Step 2. Start Qdrant and PostgreSQL in Docker** (use your own password instead of `your_password`)
```powershell
docker run -d --name proxigen-qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
docker run -d --name proxigen-postgres -e POSTGRES_PASSWORD=your_password -e POSTGRES_DB=proxigen -p 5432:5432 -v proxigen_pgdata:/var/lib/postgresql/data postgres:16
```
Check with `docker ps`: both containers should show `Up`.

Then continue with the **[shared steps](#shared-steps-for-options-2-and-3)** below (Step 3 onward).

**Next time:** start Docker Desktop, run `docker start proxigen-qdrant proxigen-postgres`, then do the "Run" steps.

---

### Option 3: Manual, PostgreSQL installed on your computer

Same as Option 2, but PostgreSQL is a normal program on your computer instead of a Docker container. Qdrant still runs in Docker.

**Install first:** everything from Option 2, plus **PostgreSQL** from https://www.postgresql.org/download/. During installation, remember the **password** you set for the `postgres` user and keep the default port **5432**.

**Step 1. Clone the project**
```powershell
git clone https://github.com/MaazZarif/proxigen_rag_project.git
cd proxigen_rag_project
```

**Step 2a. Start Qdrant in Docker** (only Qdrant, **not** the PostgreSQL container)
```powershell
docker run -d --name proxigen-qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Step 2b. Create the database in your local PostgreSQL**

Make sure the PostgreSQL service is running (on Windows it starts automatically after installation), then create an empty database called `proxigen`:
```powershell
psql -U postgres -c "CREATE DATABASE proxigen;"
```
Enter the password from the installation when asked. If `psql` is not recognized, open **SQL Shell (psql)** from the Start menu, press Enter through the prompts, enter your password, and run `CREATE DATABASE proxigen;`. You can also create it in pgAdmin.

The application creates its own tables, so no other database setup is needed.

Then continue with the **[shared steps](#shared-steps-for-options-2-and-3)** below (Step 3 onward). In the `.env` file, use **your PostgreSQL installation password** in `CHECKPOINT_DB_URL`.

**Next time:** make sure the PostgreSQL service and Docker are running, run `docker start proxigen-qdrant`, then do the "Run" steps.

---

### Shared steps for Options 2 and 3

**Step 3. Create the `.env` file**

Create a file named `.env` in the project folder (`notepad .env` on Windows, `nano .env` on Linux/macOS) and paste:

```env
GROQ_API_KEY=your_groq_api_key
CHECKPOINT_DB_URL=postgresql://postgres:your_password@localhost:5432/proxigen
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=proxigen_knowledge
```

- Replace `your_groq_api_key` with your real key.
- Replace `your_password` with the password you gave PostgreSQL (Option 2: the one in the `docker run` command; Option 3: your installation password). Use letters and numbers only; special characters would need URL-encoding.

Optional, only for LangSmith debugging (the chatbot works without it):
```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=proxigen-rag
```

**Step 4. Install the packages** (the slow step: a few minutes)
```powershell
uv sync
```

**Step 5. Load the knowledge base into Qdrant**

This also downloads the embedding model (about 550 MB) on the first run:
```powershell
uv run python -m proxigen_rag.ingestion.ingestion
```
It should end with `Successfully ingested 66 chunks.`

**Step 6. Run the backend** (terminal 1)

Keep `--reload` on Windows, otherwise the PostgreSQL driver fails to start:
```powershell
uv run uvicorn proxigen_rag.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir src
```
Wait for `Proxigen RAG API started successfully`. The first start also downloads the re-ranking model (about 90 MB). Check the backend at http://127.0.0.1:8000/health, which should show `{"status":"ok"}`.

**Step 7. Run the frontend with Live Server**

1. Open the project folder in VS Code.
2. Install the **Live Server** extension if you don't have it.
3. Open `proxigen-chat-frontend/index.html`.
4. Right-click it and choose **Open with Live Server**, or click **Go Live** at the bottom right.
5. The page opens at `http://127.0.0.1:5500/...`. Click **Chat with us** and ask a question, for example "What products does Proxigen offer?"

Live Server's default port is 5500, which is the only address the backend accepts, so don't change it. If the chat shows "Couldn't reach the chat service", check that the backend is running and that Live Server is on port 5500.