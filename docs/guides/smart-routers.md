# Smart Routers

Smart Routers use a fast LLM to intelligently route each request to the best candidate model based on the query content.

## How It Works

1. **Request arrives** with model name set to your Smart Router
2. **Designator LLM** analyzes the query and selects the best candidate
3. **Request forwards** to the selected model
4. **Response returns** to the client (with routing metadata in headers)

```
Client Request → Smart Router → Designator LLM → Best Candidate → Response
                     ↓
              "This is a coding question,
               route to Claude Sonnet"
```

## Creating a Smart Router

In the Admin UI:

1. Go to **Smart Routers** in the sidebar
2. Click **New Router**
3. Configure:

| Field | Description |
|-------|-------------|
| **Name** | The model name clients will use (e.g., `smart-router`) |
| **Description** | Optional description for the Admin UI |
| **Designator Model** | Fast model that picks the best candidate |
| **Candidates** | List of models to choose from |
| **Strategy** | `per_request` (decide each time) or `session` (sticky) |
| **Tags** | Optional tags for cost attribution |

4. Click **Save**

## Example Configuration

**Router Name**: `smart`

**Designator Model**: `gemini/gemini-2.0-flash` (fast and cheap)

**Candidates**:
- `anthropic/claude-sonnet-4-5` - Best for coding and analysis
- `openai/gpt-4o` - Good all-rounder
- `groq/llama-3.3-70b-versatile` - Fast for simple queries

**Strategy**: `per_request`

Now use it:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "smart", "messages": [{"role": "user", "content": "Write a Python function to parse JSON"}]}'
```

The designator will route coding questions to Claude, general questions to GPT-4, and simple queries to Llama.

## Designator Model Selection

Choose a designator model that's:
- **Fast** - Adds latency to every request
- **Cheap** - Called on every request
- **Good at classification** - Needs to understand query intent

Recommended designators:
- `gemini/gemini-2.0-flash` - Very fast, good reasoning
- `groq/llama-3.3-70b-versatile` - Extremely fast
- `anthropic/claude-haiku-4-5` - Good balance

Avoid using expensive models (Opus, o1) as designators.

## Routing Strategies

### per_request (Default)

The designator evaluates each request independently. Best for:
- Varied query types
- Maximum routing accuracy
- Stateless applications

### session

The designator picks a model on the first request, then sticks with it for the session. Best for:
- Conversational contexts
- Reducing designator calls
- Consistent model behavior within a conversation

Session is determined by the conversation history hash.

## Model Intelligence

Model Intelligence enriches the designator's knowledge about each candidate model by gathering comparative assessments from the web.

### Enabling Model Intelligence

1. Ensure ChromaDB is configured (`CHROMA_URL`)
2. Configure a web search provider (SearXNG recommended)
3. In the Smart Router edit modal, enable **Model Intelligence**
4. Configure:
   - **Search Provider**: SearXNG, Perplexity, or Jina
   - **Summarizer Model**: Model to summarize gathered intelligence

### How It Works

When enabled, LLM Relay:
1. Searches for reviews and comparisons of your candidate models
2. Extracts relevant assessments about strengths and weaknesses
3. Stores summaries in ChromaDB
4. Includes this context when the designator makes routing decisions

This helps the designator make more informed choices based on real-world model performance data.

### Refreshing Intelligence

Click **Refresh Intelligence** in the router's edit modal to re-gather model assessments. Do this when:
- You add new candidate models
- Significant time has passed (models improve)
- You want updated comparative data

## Viewing Routing Decisions

### Response Headers

Every response includes routing metadata:

```
X-LLM-Relay-Router: smart
X-LLM-Relay-Routed-To: anthropic/claude-sonnet-4-5
```

### Admin UI

The Dashboard and Usage pages show:
- Which router was used
- Which candidate was selected
- Designator token usage (tracked separately)

## Tips

1. **Start simple** - Begin with 2-3 candidates, add more as needed
2. **Monitor routing** - Check the Dashboard to see if routing matches expectations
3. **Tune candidates** - Remove models that are rarely selected
4. **Use descriptive names** - Help the designator with clear model descriptions in the purpose field
5. **Consider costs** - The designator adds token usage to every request

## Troubleshooting

### Router always picks the same model

- Check if your queries are too similar
- Try a smarter designator model
- Enable Model Intelligence for better context

### High latency

- Switch to a faster designator (Groq, Gemini Flash)
- Use `session` strategy to reduce designator calls
- Check designator model availability

### Designator errors

- Verify the designator model is configured and accessible
- Check provider API key is set
- Look for errors in container logs
