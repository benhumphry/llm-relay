# Smart RAGs

Smart RAGs (Retrieval-Augmented Generation) enhance LLM requests with relevant content from your document collections. Index your documents once, then every request automatically retrieves and injects relevant context.

## How It Works

1. **Documents indexed** into ChromaDB as vector embeddings
2. **Request arrives** with model name set to your Smart RAG
3. **Query embedded** using the same embedding model
4. **Semantic search** finds relevant document chunks
5. **Reranking** improves relevance ordering with cross-encoder
6. **Context injection** adds chunks to the system prompt
7. **Target model** receives the augmented request

```
Client Request → Smart RAG → Embed Query → ChromaDB Search → Rerank
                                ↓              ↓                ↓
                         nomic-embed-text  Vector Search   Cross-Encoder
                                                                ↓
                                                    Inject into System Prompt
                                                                ↓
                                                          Target Model
```

## Prerequisites

- **ChromaDB**: Required for vector storage (`CHROMA_URL` environment variable)

## Creating a Smart RAG

In the Admin UI:

1. Go to **Smart RAGs** in the sidebar
2. Click **New RAG**
3. Configure:

| Field | Description |
|-------|-------------|
| **Name** | The model name clients will use (e.g., `docs-assistant`) |
| **Target Model** | The underlying model to call with document context |
| **Source Path** | Path to your documents folder (inside container) |
| **Embedding Provider** | Local, Ollama, or cloud provider |
| **Embedding Model** | Model for creating embeddings |
| **Max Results** | Number of chunks to retrieve |
| **Similarity Threshold** | Minimum relevance score (0-1) |
| **Max Context Tokens** | Token limit for injected context |

4. Click **Save**
5. Click **Index Now** to process your documents

## Example Configuration

**RAG Name**: `docs-assistant`

**Target Model**: `anthropic/claude-sonnet-4-5`

**Source Path**: `/data/documents`

**Embedding Provider**: `ollama:winpc`

**Embedding Model**: `nomic-embed-text:latest`

**Max Results**: `5`

**Similarity Threshold**: `0.7`

**Max Context Tokens**: `4000`

Now use it:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "docs-assistant", "messages": [{"role": "user", "content": "What does the Q3 report say about revenue?"}]}'
```

The RAG will:
1. Embed your query
2. Search for relevant chunks in indexed documents
3. Rerank results for better relevance
4. Inject top chunks into Claude's context
5. Return Claude's response informed by your documents

## Document Sources

### Local Folder

Mount your document folder into the container:

```yaml
# docker-compose.yml
services:
  llm-relay:
    volumes:
      - ./my-documents:/data/documents
```

Then set Source Path to `/data/documents`.

### MCP Integrations

Smart RAGs can index documents from external services via MCP (Model Context Protocol) servers. This allows you to index content from:

- **Notion** - Workspace pages and databases
- **Google Drive** - Documents, spreadsheets, PDFs
- **GitHub** - Repository files, issues, documentation
- **Slack** - Channel messages and files
- **PostgreSQL** - Database tables

#### Setting Up Notion

1. **Create a Notion Integration**:
   - Go to [notion.so/my-integrations](https://notion.so/my-integrations)
   - Click "New integration"
   - Give it a name (e.g., "LLM Relay")
   - Select your workspace
   - Copy the "Internal Integration Secret" (starts with `ntn_`)

2. **Add the token to your environment**:
   ```bash
   # .env
   NOTION_TOKEN=ntn_your_token_here
   ```

3. **Connect the integration to your pages**:
   - Open a Notion page you want to index
   - Click the **⋯** menu (top right)
   - Select **"Add connections"**
   - Find and select your integration
   - Click **Confirm**
   
   The integration can now access that page and all its children.

4. **Create a Smart RAG with Notion source**:
   - In Admin UI, go to **Smart RAGs**
   - Click **Add RAG**
   - For **Document Source**, select **Notion** from the MCP Integrations group
   - Configure target model and other settings
   - Click **Save**, then **Index Now**

#### Setting Up Google (Drive/Gmail/Calendar)

Index content from Google Drive, Gmail, or Calendar using OAuth authentication.

1. **Create OAuth Credentials**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create or select a project
   - Enable the APIs you need: **Drive API**, **Gmail API**, and/or **Calendar API**
   - Go to **APIs & Services** → **Credentials**
   - Click **Create Credentials** → **OAuth client ID**
   - Application type: **Web application**
   - Add authorized redirect URI: `http://your-server:8080/oauth/google/callback`
   - Note the **Client ID** and **Client Secret**

2. **Configure OAuth consent screen**:
   - Go to **APIs & Services** → **OAuth consent screen**
   - Configure the consent screen with your app name
   - Add scopes for services you want to use:
     - `drive.readonly` - for Google Drive
     - `gmail.readonly` - for Gmail
     - `calendar.readonly` - for Calendar
   - Add test users (your email addresses) if in testing mode

3. **Add environment variables**:
   ```bash
   # .env
   GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-client-secret
   OAUTH_ENCRYPTION_KEY=your-32-byte-fernet-key
   ```
   
   Generate an encryption key for secure token storage:
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

4. **Create a Smart RAG with Google source**:
   - In Admin UI, go to **Smart RAGs** → **Add RAG**
   - Select **Google (Drive/Gmail/Calendar)** as the source
   - Click **Connect** to link your Google account
   - Complete the OAuth flow in the popup
   - Select the connected account from the dropdown
   - Choose which service to index: **Drive**, **Gmail**, or **Calendar**
   - Configure target model and save

5. **Index multiple services** (optional):
   - Create separate RAGs for each service you want to index
   - Each RAG can use the same Google account but index a different service
   - Or connect multiple accounts to index from different users

#### Setting Up GitHub

1. **Create a GitHub Personal Access Token**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate a token with `repo` scope

2. **Add to environment**:
   ```bash
   # .env
   GITHUB_TOKEN=ghp_your_token_here
   ```

3. **Create a Smart RAG** with the GitHub source preset.

#### Setting Up Slack

1. **Create a Slack App**:
   - Go to [api.slack.com/apps](https://api.slack.com/apps)
   - Create a new app and add Bot Token Scopes for channel reading

2. **Add to environment**:
   ```bash
   # .env
   SLACK_BOT_TOKEN=xoxb-your-token
   SLACK_TEAM_ID=T0123456789
   ```

3. **Create a Smart RAG** with the Slack source preset.

#### Custom MCP Servers

For other MCP-compatible servers:

1. Select **Custom MCP Server** as the source
2. Configure:
   - **Server Name**: Identifier for the server
   - **Transport**: `stdio` (subprocess) or `http`
   - **Command/URL**: How to connect to the server
   - **Arguments**: Command-line arguments (for stdio)

### Supported Formats

LLM Relay uses Docling for document parsing:

- **PDF** - Full text extraction with layout preservation
- **DOCX** - Microsoft Word documents
- **PPTX** - PowerPoint presentations
- **HTML** - Web pages
- **Markdown** - `.md` files
- **Images** - PNG, JPG (requires vision model)

## Embedding Providers

### Local (Default)

Uses bundled sentence-transformers. No external dependencies.

**Pros**: Fast, free, private
**Cons**: Less accurate than larger models

### Ollama

Uses your Ollama instance for embeddings.

**Setup**:
```bash
# On your Ollama instance
ollama pull nomic-embed-text
```

**Configuration**:
- Embedding Provider: `ollama:<instance-name>`
- Embedding Model: `nomic-embed-text:latest`
- Ollama URL: `http://your-ollama:11434`

**Pros**: Good balance of quality and speed
**Cons**: Requires Ollama instance

### Cloud Providers

Use any configured provider with embedding support (OpenAI, etc.):

- Embedding Provider: `openai`
- Embedding Model: `text-embedding-3-small`

**Pros**: High quality embeddings
**Cons**: API costs, data leaves your network

## Vision Model for PDFs

For complex PDFs with tables, charts, or images, configure a vision model to improve parsing:

1. In Admin UI, go to **Settings**
2. In **Web Search & Scraping** section, configure:
   - Vision Provider: `ollama` or cloud provider
   - Vision Model: `granite3.2-vision:latest` or similar

The vision model will be used during indexing to extract content from complex document pages.

## Reranking

Smart RAGs automatically rerank retrieved chunks using a cross-encoder model. This significantly improves relevance by comparing query-document pairs directly.

**Default Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~48MB)

The reranker:
1. Takes initial results from vector search
2. Scores each chunk against the query
3. Reorders by relevance
4. Returns top results

### Jina Reranker (Optional)

For API-based reranking:
- Rerank Provider: `jina`
- Set `JINA_API_KEY` environment variable

## Indexing

### Manual Indexing

1. Go to **Smart RAGs** in Admin UI
2. Click on your RAG
3. Click **Index Now**

Indexing processes all documents in the source folder and creates embeddings.

### Scheduled Indexing

Configure automatic re-indexing with cron syntax:

| Schedule | Cron Expression |
|----------|-----------------|
| Every hour | `0 * * * *` |
| Daily at midnight | `0 0 * * *` |
| Weekly on Sunday | `0 0 * * 0` |

Set the **Index Schedule** field in the RAG configuration.

### Index Status

| Status | Meaning |
|--------|---------|
| `ready` | Indexed and available |
| `indexing` | Currently processing |
| `error` | Indexing failed (check logs) |
| `pending` | Not yet indexed |

## Configuration Options

### Chunk Size and Overlap

Control how documents are split:

- **Chunk Size**: Characters per chunk (default: 512)
- **Chunk Overlap**: Overlap between chunks (default: 50)

Smaller chunks = more precise retrieval but less context per chunk.
Larger chunks = more context but may include irrelevant content.

### Similarity Threshold

Minimum relevance score for chunks to be included.

| Value | Behavior |
|-------|----------|
| `0.9` | Only very relevant chunks |
| `0.7` | Moderately relevant (recommended) |
| `0.5` | Loosely relevant |

Start with 0.7 and adjust based on retrieval quality.

### Max Results

Number of chunks to retrieve and inject.

| Use Case | Recommended |
|----------|-------------|
| Quick answers | 3-5 |
| Detailed research | 5-10 |
| Comprehensive context | 10-15 |

More chunks = more context but higher token usage.

## Context Injection

Retrieved chunks are injected into the system prompt:

```xml
<document_context>
The following information was retrieved from the document collection to help answer the user's question.
Use this information to provide an accurate response. Cite the source files when relevant.

## Relevant Document Context

[Source: quarterly-report-q3.pdf]
Revenue increased 15% year-over-year to $2.3M...

---

[Source: financial-summary.docx]
The Q3 results exceeded analyst expectations...

</document_context>
```

## Viewing Retrieval

### Response Headers

```
X-LLM-Relay-RAG: docs-assistant
X-LLM-Relay-Chunks: 5
X-LLM-Relay-Sources: quarterly-report-q3.pdf, financial-summary.docx
```

### Admin UI

The Smart RAG detail page shows:
- Total requests
- Context injection rate
- Document count
- Chunk count
- Index status

## Best Practices

1. **Organize documents** - Keep related documents together
2. **Use descriptive filenames** - Helps with source attribution
3. **Start with default settings** - Tune after testing
4. **Monitor injection rate** - Low rate may indicate threshold too high
5. **Re-index when documents change** - Or use scheduled indexing
6. **Use vision model for PDFs** - Improves table/chart extraction

## Troubleshooting

### No chunks retrieved

- Lower the similarity threshold (try 0.5)
- Check if documents were indexed successfully
- Verify embedding model is working
- Check ChromaDB connection

### Wrong or irrelevant chunks

- Raise the similarity threshold
- Reduce chunk size for more precise matching
- Check if documents contain the expected content

### Indexing fails

- Check container logs for errors
- Verify source path is accessible
- Ensure embedding model is available
- Check ChromaDB is running

### High latency

- Reduce max_results
- Use local embedding model
- Check ChromaDB performance

### Large documents not indexed

- Docling may timeout on very large files
- Split into smaller documents
- Check container memory limits
