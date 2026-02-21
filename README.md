# Code Horde

Self-hosted multi-agent AI platform for agentic development and security operations.

---

## Quick Start (macOS)

### Prerequisites

| Tool | Install |
|---|---|
| **Python 3.12+** | `brew install python@3.12` |
| **Docker Desktop** | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| **Ollama** | `brew install ollama` |
| **ngrok** | `brew install ngrok` (free account at [ngrok.com](https://ngrok.com)) |

### 1. Setup

```bash
cd code-horde
make setup
```

This creates a virtual environment, installs dependencies, starts infrastructure
(Redis, PostgreSQL via Homebrew or Docker, Neo4j via Docker), generates Ed25519
security keys, and creates `.env`.

### 2. Configure

Edit `.env` with your real credentials:

```bash
AGENTARMY_CLAUDE_API_KEY=sk-ant-...        # Anthropic API key
AGENTARMY_WHATSAPP_API_TOKEN=...           # Meta Developer Console
AGENTARMY_WHATSAPP_PHONE_NUMBER_ID=...     # WhatsApp Business Account
AGENTARMY_WHATSAPP_VERIFY_TOKEN=...        # Any random string
GITHUB_TOKEN=ghp_...                        # GitHub Personal Access Token
```

### 3. Start Ollama

```bash
ollama serve                    # Terminal 1
ollama pull qwen2.5-coder:7b   # Terminal 2 (one-time)
```

### 4. Run

```bash
make dev     # Starts app with hot-reload on :8000
```

The Command Center dashboard opens at `http://localhost:8000`.

### 5. WhatsApp Webhooks (optional)

```bash
make tunnel  # Opens ngrok tunnel
```

Copy the ngrok URL into Meta Developer Console:
**WhatsApp → Configuration → Webhook URL** → `https://xxxxx.ngrok-free.app/webhook/whatsapp`

### 6. Verify

```bash
make status
curl http://localhost:8000/health
curl http://localhost:8000/agents
```

---

## Running Modes

Code Horde can be started in three ways depending on the context:

**Development (hot-reload)** — recommended for daily work:

```bash
make dev
```

Runs `uvicorn` with `--reload` on port 8000. Auto-restarts on code changes in `src/`.

**Full stack (infra + app)** — starts Docker services and then the app in one go:

```bash
make up
```

Equivalent to `make infra-up` followed by `make dev`.

**Production mode (no reload, multi-worker):**

```bash
make run
```

Starts `uvicorn` with 2 workers and `--log-level info`. No hot-reload.

**macOS desktop app** — standalone native wrapper:

```bash
make app           # Build Code Horde.app in dist/
make app-install   # Build + copy to /Applications
make app-open      # Launch the desktop app (standalone mode)
```

**Stop everything:**

```bash
make down
```

Stops the app process and all Docker infrastructure.

---

## Architecture (macOS Local)

```
macOS MacBook
├── Docker Desktop (or Homebrew)
│   ├── Redis       (:6379)    — pub/sub, cache
│   ├── PostgreSQL  (:5432)    — state, audit trail
│   ├── RabbitMQ    (:5672)    — durable task queues (AMQP)
│   └── Neo4j       (:7687)    — knowledge graph (optional)
│
├── Python (native)
│   ├── FastAPI Gateway   (:8000)
│   ├── Commander   (orchestrator)
│   ├── Sentinel    (security)
│   ├── Builder     (development)
│   ├── Inspector   (QA)
│   ├── Watcher     (monitoring)
│   ├── Scout       (research + web)
│   ├── Scribe      (documentation)
│   ├── DevOps      (infrastructure)
│   ├── Marketer    (content & campaigns)
│   ├── Designer    (UI/UX)
│   ├── Linter      (code quality)
│   ├── Automator   (workflows)
│   └── Scheduler   (cron & scheduling)
│
├── Ollama (native, Apple Silicon GPU)
│   └── qwen2.5-coder:7b  (:11434)
│
└── ngrok → WhatsApp Cloud API → localhost:8000
```

**Why this split?** Infrastructure in Docker for isolation and easy reset. Python runs natively for instant hot-reload and debugger support. Ollama runs natively to leverage Apple Silicon GPU (Docker has no GPU passthrough on macOS). Redis and PostgreSQL prefer Homebrew when available (faster, no Docker overhead) with automatic Docker fallback.

---

## Make Commands

### Core

| Command | Description |
|---|---|
| `make setup` | First-time setup (venv, deps, infra, keys, .env) |
| `make dev` | Start app with hot-reload (port 8000) |
| `make up` | Start infrastructure + app together |
| `make down` | Stop everything (app + infra) |
| `make run` | Start app in production mode (2 workers, no reload) |

### Desktop App

| Command | Description |
|---|---|
| `make app` | Build macOS `.app` bundle in `dist/` |
| `make app-install` | Build + install to `/Applications` |
| `make app-open` | Launch the desktop app (standalone mode) |
| `make dashboard` | Open Command Center in default browser |

### Infrastructure

| Command | Description |
|---|---|
| `make infra-up` | Start Redis, PostgreSQL, Neo4j |
| `make infra-down` | Stop infrastructure services |
| `make infra-status` | Show port status and service health |
| `make db-reset` | Reset database (destroys all data) |
| `make ollama-setup` | Install Ollama + pull recommended models |
| `make ollama-models` | List installed Ollama models |

### CLI & Monitoring

| Command | Description |
|---|---|
| `make cli` | Interactive terminal (connects to running API) |
| `make cli-exec CMD="/status"` | One-shot CLI command |
| `make status` | Health check all components |
| `make logs` | Tail Docker infrastructure logs |
| `make tunnel` | ngrok tunnel for WhatsApp webhooks |

### Quality & Security

| Command | Description |
|---|---|
| `make test` | Run test suite |
| `make lint` | Run ruff linter |
| `make format` | Format code (black + isort) |
| `make check` | Lint + mypy type checking |
| `make keys` | Regenerate JWT secret + Ed25519 agent keys |
| `make clean` | Remove all generated data (venv, keys, volumes) |

---

## WhatsApp Commands

| Command | Description |
|---|---|
| `/status` | System health overview |
| `/agents` | List active agents |
| `/task <desc>` | Create a new task |
| `/deploy <env>` | Trigger deployment (requires approval) |
| `/security` | Latest security scan results |
| `/logs <agent>` | View agent logs |
| `/approve <id>` | Approve pending operation |
| `/reject <id>` | Reject pending operation |
| `/cost` | API usage and costs |

Natural language is also supported.

---

## Security

- **Zero-Trust**: every action authenticated, authorized, audited
- **Capability-based permissions**: YAML manifests per agent
- **Immutable audit trail**: hash-chained log entries
- **Ed25519 signatures**: asymmetric signing for inter-agent messages (HMAC-SHA256 fallback)
- **Multi-model routing**: sensitive data → local Ollama, complex tasks → Claude
- **Skill registry**: centralized, versioned skill management with REST API

Full details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Production (Linux)

```bash
docker compose -f docker/docker-compose.yaml up -d
```

---

## License

MIT
