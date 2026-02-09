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
(Redis, PostgreSQL, RabbitMQ) in Docker, generates security keys, and creates `.env`.

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

### 5. WhatsApp Webhooks

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

## Architecture (macOS Local)

```
macOS MacBook
├── Docker Desktop
│   ├── Redis       (:6379)    — message bus, cache
│   ├── PostgreSQL  (:5432)    — state, audit trail
│   └── RabbitMQ    (:5672)    — durable task queues
│
├── Python (native)
│   ├── FastAPI Gateway  (:8000)
│   ├── Commander (orchestrator)
│   ├── Sentinel  (security)
│   ├── Builder   (development)
│   ├── Inspector (QA)
│   ├── Watcher   (monitoring)
│   ├── Scout     (research)
│   ├── Scribe    (documentation)
│   └── DevOps    (infrastructure)
│
├── Ollama (native, Apple Silicon GPU)
│   └── qwen2.5-coder:7b  (:11434)
│
└── ngrok → WhatsApp Cloud API → localhost:8000
```

**Why this split?** Infrastructure in Docker for isolation and easy reset. Python runs natively for instant hot-reload and debugger support. Ollama runs natively to leverage Apple Silicon GPU (Docker has no GPU passthrough on macOS).

---

## Commands

| Command | Description |
|---|---|
| `make setup` | First-time setup |
| `make dev` | Start app with hot-reload |
| `make up` | Start infrastructure + app |
| `make down` | Stop everything |
| `make tunnel` | ngrok tunnel for webhooks |
| `make status` | Health check all components |
| `make logs` | Tail Docker logs |
| `make test` | Run test suite |
| `make keys` | Regenerate security keys |
| `make ollama-setup` | Install Ollama + pull models |
| `make db-reset` | Reset database |
| `make clean` | Remove all generated data |

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
- **Ed25519 signatures**: signed inter-agent messages
- **Multi-model routing**: sensitive data → local Ollama, complex tasks → Claude

Full details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Production (Linux)

```bash
docker compose -f docker/docker-compose.yaml up -d
```

---

## License

MIT
