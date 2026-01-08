# LLM Relay User Guide

## Overview

LLM Relay is a self-hosted proxy that unifies multiple LLM providers behind Ollama and OpenAI-compatible APIs. It provides usage tracking, cost attribution, and intelligent routing features.

## Key Features

### Model Aliases
Create friendly names for models. For example, you can create an alias called "claude" that points to "anthropic/claude-sonnet-4-20250514".

### Smart Routers
Use a designator LLM to intelligently route requests to the best model based on query content. Configure candidate models and routing criteria.

### Smart Cache
Cache responses to reduce costs and latency for repeated queries. Uses semantic similarity matching via ChromaDB.

### Smart Augmentors
Enhance LLM requests with web search results or scraped content. Supports SearXNG for privacy-focused web search.

### Smart RAGs
Index local documents and retrieve relevant context to augment LLM requests. Supports PDF, DOCX, Markdown, and other file formats.

## API Endpoints

### Ollama-Compatible
- `POST /api/chat` - Chat completion
- `POST /api/generate` - Text generation
- `GET /api/tags` - List available models

### OpenAI-Compatible
- `POST /v1/chat/completions` - Chat completion
- `GET /v1/models` - List available models

## Configuration

### Environment Variables
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENAI_API_KEY` - OpenAI API key
- `GOOGLE_API_KEY` - Google AI API key
- `OLLAMA_URL` - URL to Ollama instance
- `CHROMA_URL` - URL to ChromaDB instance
- `DATABASE_URL` - PostgreSQL connection string

## Usage Tracking

All requests are logged with:
- Token counts (input, output, cached)
- Response time
- Cost calculation
- Tag attribution from API keys

Tags can be embedded in API keys using the format: `tag:project-name:actual-api-key`

## Admin Interface

Access the admin UI at port 8080 to:
- View usage statistics and costs
- Manage model aliases
- Configure smart routers and caches
- Monitor request logs
