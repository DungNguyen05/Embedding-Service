# Vietnamese Embedding Service

A high-performance, OpenAI-compatible embedding API service optimized for Vietnamese language processing. This service provides semantic embeddings using state-of-the-art multilingual models with special optimization for Vietnamese text.

## Features

- 🇻🇳 **Vietnamese Language Optimization**: Specialized preprocessing and models for Vietnamese text
- 🔄 **OpenAI API Compatibility**: Drop-in replacement for OpenAI's embedding API
- ⚡ **High Performance**: Optimized for batch processing and low latency
- 🐳 **Docker Ready**: Easy deployment with Docker and Docker Compose
- 📊 **Multiple Model Support**: Choose from different models based on your performance/accuracy needs
- 🔧 **Configurable**: Environment-based configuration for different deployment scenarios

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and setup:**
```bash
git clone <repository-url>
cd vietnamese-embedding-service
cp .env.example .env
```

2. **Configure environment (optional):**
Edit `.env` file to customize model and settings:
```bash
# Choose your model based on requirements:
# keepitreal/vietnamese-sbert - Best for Vietnamese (2GB RAM)
# sentence-transformers/LaBSE - Good multilingual (1.5GB RAM)
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 - Lightweight (500MB RAM)
EMBEDDING_MODEL_NAME=keepitreal/vietnamese-sbert
DEVICE=auto
```

3. **Start the service:**
```bash
docker-compose up -d
```

4. **Verify it's running:**
```bash
curl http://localhost:8000/health
```

### Using Docker

```bash
docker build -t vietnamese-embedding-service .
docker run -p 8000:8000 -e EMBEDDING_MODEL_NAME=keepitreal/vietnamese-sbert vietnamese-embedding-service
```

### Manual Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the service:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Usage

The service is fully compatible with OpenAI's embedding API format.

### Create Embeddings

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Tôi yêu Việt Nam",
    "model": "text-embedding-ada-002"
  }'
```

### Batch Embeddings

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "Xin chào, bạn khỏe không?",
      "Hôm nay thời tiết đẹp quá!",
      "Tôi đang học tiếng Việt"
    ],
    "model": "text-embedding-ada-002"
  }'
```

### List Models

```bash
curl "http://localhost:8000/v1/models"
```

## Integration with Mattermost Plugin

To use this service with your Mattermost plugin, update your configuration:

```json
{
  "embeddingSearchConfig": {
    "type": "composite",
    "embeddingProvider": {
      "type": "openai-compatible",
      "parameters": {
        "apiURL": "http://localhost:8000/v1",
        "apiKey": "not-needed",
        "embeddingModel": "text-embedding-ada-002",
        "embeddingDimensions": 768
      }
    },
    "vectorStore": {
      "type": "pgvector",
      "parameters": {
        "dimensions": 768
      }
    }
  }
}
```

## Model Options

| Model | Size | RAM Usage | Vietnamese Performance | Speed |
|-------|------|-----------|----------------------|-------|
| `keepitreal/vietnamese-sbert` | ~800MB | ~2GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `sentence-transformers/LaBSE` | ~500MB | ~1.5GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `paraphrase-multilingual-mpnet-base-v2` | ~420MB | ~1GB | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~120MB | ~500MB | ⭐⭐⭐ | ⭐⭐