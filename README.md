# Vietnamese Embedding Service

OpenAI-compatible embedding API service optimized for Vietnamese language processing.

## Available Models

| Model Name | Dimensions | RAM Usage | Best For |
|------------|------------|-----------|----------|
| `keepitreal/vietnamese-sbert` | 768 | ~2GB | Vietnamese text (recommended) |
| `sentence-transformers/LaBSE` | 768 | ~1.5GB | Multilingual |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | ~1GB | Multilingual balance |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~500MB | Lightweight |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~300MB | Very lightweight |

## Docker Deployment

1. **Setup:**
```bash
cp .env.example .env
# Edit .env with your preferred model
```

2. **Start service:**
```bash
docker-compose up -d
```

3. **Test:**
```bash
curl http://localhost:8000/health
```

## Manual Installation

1. **Install:**
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. **Configure:**
```bash
cp .env.example .env
# Edit .env file
```

3. **Run:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Usage

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Tôi yêu Việt Nam",
    "model": "text-embedding-ada-002"
  }'
```

**Note:** Match `embeddingDimensions` with your model's dimensions from the table above.