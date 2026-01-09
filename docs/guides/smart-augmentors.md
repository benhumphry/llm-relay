# Smart Augmentors

Smart Augmentors enhance LLM requests with real-time web content. Every request is automatically augmented with search results and scraped web pages, giving the LLM access to current information.

## How It Works

1. **Request arrives** with model name set to your Smart Augmentor
2. **Designator LLM** generates an optimized search query
3. **Web search** finds relevant results
4. **Reranking** orders URLs by relevance to the query
5. **Scraping** extracts content from top URLs
6. **Context injection** adds the content to the system prompt
7. **Target model** receives the augmented request

```
Client Request → Augmentor → Designator → Search → Rerank → Scrape
                                ↓            ↓        ↓         ↓
                         "AI news 2024"  SearXNG  Cross-Encoder  Trafilatura
                                                        ↓
                                              Inject into System Prompt
                                                        ↓
                                                  Target Model
```

## Prerequisites

- **Search Provider**: SearXNG (self-hosted), Perplexity, or Jina (requires API key)

## Creating a Smart Augmentor

In the Admin UI:

1. Go to **Smart Augmentors** in the sidebar
2. Click **New Augmentor**
3. Configure:

| Field | Description |
|-------|-------------|
| **Name** | The model name clients will use (e.g., `search-claude`) |
| **Target Model** | The underlying model to call with augmented context |
| **Designator Model** | Fast model that generates search queries |
| **Search Provider** | SearXNG, Perplexity, or Jina |
| **Max Search Results** | Number of search results to fetch |
| **Max Scrape URLs** | Number of URLs to scrape for full content |
| **Max Context Tokens** | Token limit for injected context |
| **Scraper Provider** | Built-in (trafilatura) or Jina Reader |

4. Click **Save**

## Example Configuration

**Augmentor Name**: `search-claude`

**Target Model**: `anthropic/claude-sonnet-4-5`

**Designator Model**: `gemini/gemini-2.0-flash`

**Search Provider**: `searxng`

**Max Search Results**: `6`

**Max Scrape URLs**: `3`

**Max Context Tokens**: `8000`

Now use it:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "search-claude", "messages": [{"role": "user", "content": "What happened in AI this week?"}]}'
```

The augmentor will:
1. Generate query: "AI news this week January 2025"
2. Search and get 6 results
3. Rerank URLs by relevance
4. Scrape top 3 URLs
5. Inject content into Claude's context
6. Return Claude's response with current information

## Search Providers

### SearXNG (Recommended)

Self-hosted metasearch engine. No API key required, full privacy.

**Setup**:
```yaml
# docker-compose.yml
services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8888:8080"
```

**Configuration**:
- Search Provider: `searxng`
- Search Provider URL: `http://searxng:8080` (or leave empty for env var)

Set `SEARXNG_URL` in your environment.

### Perplexity

Uses Perplexity's API for search. Requires API key.

**Configuration**:
- Search Provider: `perplexity`
- Set `PERPLEXITY_API_KEY` in environment

### Jina Search

Uses Jina's search API. Requires API key.

**Configuration**:
- Search Provider: `jina`
- Set `JINA_API_KEY` in environment

## Scraper Providers

### Built-in (Default)

Uses httpx for fetching and trafilatura for content extraction. Works well for most sites.

**Pros**: No external dependencies, fast
**Cons**: May fail on JavaScript-heavy sites

### Jina Reader

Uses Jina's Reader API (`r.jina.ai`) for content extraction. Handles JavaScript rendering.

**Configuration**:
- Scraper Provider: `jina`
- Optionally set `JINA_API_KEY` for higher rate limits

**Pros**: Handles JavaScript, clean markdown output
**Cons**: External API dependency, rate limits without API key

## Reranking

Smart Augmentors automatically rerank search results before scraping using a cross-encoder model. This ensures the most relevant URLs are scraped first.

**Default Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~48MB, runs locally)

The reranker scores each URL based on how well its title and snippet match the search query, then selects the top URLs for scraping.

### Jina Reranker (Optional)

For API-based reranking, configure:
- Rerank Provider: `jina`
- Set `JINA_API_KEY` in environment

## Context Injection

Augmented content is injected into the system prompt:

```xml
<augmented_context>
Today's date: 2025-01-15

The following information was retrieved from the web to help answer the user's question.
Use this information to provide an accurate, up-to-date response.

## Web Search Results

### 1. AI Breakthroughs in 2025
URL: https://example.com/ai-news
Summary of the article...

---

## Scraped Content

### From: https://example.com/ai-news
Full article content here...

</augmented_context>
```

The target model sees this context and can use it to provide informed responses.

## Configuration Tips

### Designator Model

Choose a fast, cheap model:
- `gemini/gemini-2.0-flash` - Very fast
- `groq/llama-3.3-70b-versatile` - Extremely fast
- `anthropic/claude-haiku-4-5` - Good balance

The designator generates search queries, not final responses.

### Search vs Scrape Balance

| Use Case | Search Results | Scrape URLs |
|----------|----------------|-------------|
| Quick answers | 6 | 2 |
| Research | 10 | 5 |
| Deep dive | 15 | 8 |

More scraping = better context but higher latency.

### Token Limits

Set `max_context_tokens` based on your target model's context window:
- 4000-8000 for most queries
- 16000+ for deep research

## Viewing Augmentation

### Response Headers

```
X-LLM-Relay-Augmentor: search-claude
X-LLM-Relay-Augmentation: search+scrape
X-LLM-Relay-Search-Query: AI news January 2025
```

### Admin UI

The Dashboard shows:
- Augmentation rate (% of requests augmented)
- Search requests
- Scrape requests
- Designator token usage

## Best Practices

1. **Use fast designators** - They're called on every request
2. **Limit scraping** - 2-3 URLs is usually sufficient
3. **Match token limits** - Don't exceed target model's context
4. **Monitor latency** - Augmentation adds 2-5 seconds typically
5. **Use SearXNG** - Self-hosted, no rate limits, private

## Troubleshooting

### No search results

- Verify search provider is configured
- Check `SEARXNG_URL` or provider API key
- Look for errors in container logs

### Scraping failures

- Some sites block automated access (403/401 errors)
- Try Jina Reader for JavaScript-heavy sites
- Check container logs for specific errors

### High latency

- Reduce `max_scrape_urls`
- Use a faster designator model
- Consider caching with Smart Cache

### Poor search queries

- The designator generates queries from user input
- Try a smarter designator model
- Check if user queries are clear enough

### Context too large

- Reduce `max_context_tokens`
- Reduce `max_scrape_urls`
- Some scraped pages may be very long
