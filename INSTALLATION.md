# Installation Guide

This guide covers all installation options for LLM Relay, from simple single-container setups to full production deployments with all optional services.

## Quick Start (Simplest)

For basic usage with SQLite storage:

```bash
git clone https://github.com/benhumphry/llm-relay.git
cd llm-relay
cp .env.example .env
# Edit .env and add your API keys

docker compose up -d
```

Your proxy is now running:
- **API**: http://localhost:11434 (Ollama/OpenAI compatible)
- **Admin UI**: http://localhost:8080

---

## Installation Options

| Setup | Database | Features | Best For |
|-------|----------|----------|----------|
| [Basic](#basic-setup) | SQLite | Core proxy, tracking, aliases, smart routers | Personal use, testing |
| [PostgreSQL](#postgresql-setup) | PostgreSQL | Same as Basic + better concurrency | Production, multi-user |
| [Full Stack](#full-stack-setup) | PostgreSQL | All features including Cache, RAG, Web, Memory | Full feature set |

---

## Basic Setup

Uses SQLite for storage. Simple and requires no additional services.

```bash
docker compose up -d
```

### Configuration

Create `.env` from the example:

```bash
cp .env.example .env
```

Add your provider API keys:

```env
# Required: At least one provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Optional: Additional providers
GROQ_API_KEY=...
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...

# Optional: Admin password (random if not set)
ADMIN_PASSWORD=your-secure-password
```

---

## PostgreSQL Setup

Recommended for production deployments. Better concurrent write performance and easier backups.

```bash
docker compose -f docker-compose.postgresql.yml up -d
```

Or using the full stack file:

```bash
docker compose -f docker-compose.full.yml up -d llm-relay postgres
```

### Using an Existing PostgreSQL Server

If you have an existing PostgreSQL server, modify `.env`:

```env
DATABASE_URL=postgresql://user:password@hostname:5432/llmrelay
```

Then use the basic docker-compose.yml (no need for the postgres service).

---

## Full Stack Setup

Includes all optional services for maximum functionality:

| Service | Purpose | Required For |
|---------|---------|--------------|
| PostgreSQL | Database | Production deployments |
| ChromaDB | Vector storage | Smart Alias features (Cache, RAG, Web), Model Intelligence |
| SearXNG | Web search | Smart Alias Web enrichment |

### Start All Services

```bash
docker compose -f docker-compose.full.yml up -d
```

### Start Specific Services

```bash
# Core + PostgreSQL + ChromaDB (no web search)
docker compose -f docker-compose.full.yml up -d llm-relay postgres chroma

# Core + PostgreSQL only
docker compose -f docker-compose.full.yml up -d llm-relay postgres
```

### Environment Variables

The full stack adds these optional variables:

```env
# ChromaDB connection (enables Smart Cache/Augmentor/RAG)
CHROMA_URL=http://chroma:8000
CHROMA_COLLECTION_PREFIX=llmrelay_

# SearXNG connection (enables web search in Smart Augmentor)
SEARXNG_URL=http://searxng:8080
```

---

## Service Details

### ChromaDB

[ChromaDB](https://www.trychroma.com/) is an open-source vector database used for:

- **Semantic Caching**: Returns cached answers for similar queries (Smart Alias with Cache enabled)
- **RAG**: Stores document embeddings for semantic retrieval (Smart Alias with RAG enabled)
- **Web Enrichment**: Caches web search results and scraped content (Smart Alias with Web enabled)
- **Model Intelligence**: Caches comparative model assessments for smart routing

**Resource requirements**: ~500MB RAM minimum, more for large collections

**Data persistence**: Stored in `chroma-data` volume

**Health check**: http://localhost:8000/api/v2/heartbeat

### SearXNG

[SearXNG](https://docs.searxng.org/) is a privacy-respecting metasearch engine used for:

- **Web Enrichment**: Web search to inject current information into LLM context (Smart Alias with Web enabled)
- **Model Intelligence**: Searching for model reviews and benchmarks

**Alternative**: You can use Perplexity API or Jina Search instead of SearXNG by setting `PERPLEXITY_API_KEY` or `JINA_API_KEY` and configuring the search provider in Web Config.

**First-time setup**:
1. Start SearXNG: `docker compose -f docker-compose.full.yml up -d searxng`
2. Verify at http://localhost:8888
3. Optionally customize `./searxng/settings.yml`

**Note**: The default configuration exposes port 8888 for testing. In production, remove the port mapping and access SearXNG only through the internal Docker network.

---

## Docker Compose Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Basic setup with SQLite |
| `docker-compose.postgresql.yml` | Core + PostgreSQL |
| `docker-compose.full.yml` | All services (PostgreSQL, ChromaDB, SearXNG) |
| `docker-compose.swarm.yml` | Docker Swarm deployment |

---

## Environment Variables Reference

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `11434` | API server port |
| `HOST` | `0.0.0.0` | API server bind address |
| `ADMIN_PORT` | `8080` | Admin UI port |
| `ADMIN_HOST` | `0.0.0.0` | Admin UI bind address |
| `ADMIN_PASSWORD` | (random) | Admin UI password |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///data/llm_relay.db` | Database connection URL |

### Provider API Keys

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `OPENAI_API_KEY` | OpenAI (GPT) |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `MISTRAL_API_KEY` | Mistral |
| `GROQ_API_KEY` | Groq |
| `XAI_API_KEY` | xAI (Grok) |
| `PERPLEXITY_API_KEY` | Perplexity |
| `OPENROUTER_API_KEY` | OpenRouter |
| `FIREWORKS_API_KEY` | Fireworks |
| `TOGETHER_API_KEY` | Together AI |
| `DEEPINFRA_API_KEY` | DeepInfra |
| `CEREBRAS_API_KEY` | Cerebras |
| `SAMBANOVA_API_KEY` | SambaNova |
| `COHERE_API_KEY` | Cohere |

API keys support Docker secrets via `_FILE` suffix:
```yaml
environment:
  - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
```

### Optional Services

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_URL` | (none) | ChromaDB server URL |
| `CHROMA_COLLECTION_PREFIX` | `llmrelay_` | Prefix for ChromaDB collections |
| `SEARXNG_URL` | (none) | SearXNG server URL |

---

## Upgrading

### Standard Upgrade

```bash
docker compose pull
docker compose up -d
```

### Database Migrations

Migrations run automatically on startup. No manual steps required.

### Breaking Changes

Check the [CHANGELOG](CHANGELOG.md) for any breaking changes between versions.

---

## Troubleshooting

### Container won't start

Check logs:
```bash
docker compose logs llm-relay
```

Common issues:
- Missing required API keys in `.env`
- Port already in use (change `PORT` or `ADMIN_PORT`)
- Database connection failed (check `DATABASE_URL`)

### ChromaDB not connecting

1. Verify ChromaDB is running: `docker compose ps chroma`
2. Check health: `curl http://localhost:8000/api/v2/heartbeat`
3. Verify `CHROMA_URL` is set correctly
4. Check Admin UI: `/api/chroma/status` endpoint

### SearXNG not working

1. Verify SearXNG is running: `docker compose ps searxng`
2. Test search: http://localhost:8888
3. Check logs: `docker compose logs searxng`

### Performance issues

- **High memory usage**: ChromaDB caches embeddings; increase container memory limits
- **Slow responses**: Check if designator model (for smart routers) is overloaded
- **Database locks**: Switch from SQLite to PostgreSQL for concurrent access

---

## Production Recommendations

1. **Use PostgreSQL** for better concurrency and easier backups
2. **Set a strong `ADMIN_PASSWORD`** instead of using the random default
3. **Use Docker secrets** for API keys instead of environment variables
4. **Remove exposed ports** for internal services (ChromaDB, SearXNG)
5. **Set up backups** for PostgreSQL and ChromaDB data volumes
6. **Use a reverse proxy** (nginx, Traefik) for HTTPS termination
7. **Monitor disk usage** - request logs can grow large over time

---

## Without Docker

For development or non-Docker deployments:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install ChromaDB client
pip install chromadb

# Run
python proxy.py
```

Set environment variables directly or use a `.env` file with `python-dotenv`.
