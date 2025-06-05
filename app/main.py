from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
from app.api.embeddings import router as embeddings_router

# Load environment variables
load_dotenv()

# Configure logging from environment variable
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vietnamese Embedding Service",
    description="OpenAI-compatible embedding API",
    version="1.0.0",
    docs_url=None,  # Disable docs
    redoc_url=None  # Disable redoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(embeddings_router, tags=["embeddings"])

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        from app.models.embedding_model import get_embedding_model
        get_embedding_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)