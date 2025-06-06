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
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vietnamese Embedding Service",
    description="OpenAI-compatible embedding API optimized for Vietnamese language",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None
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
        logger.info("Starting Vietnamese Embedding Service...")
        from app.models.embedding_model import get_embedding_model
        model = get_embedding_model()
        logger.info(f"Model loaded successfully: {model.get_model_name()}")
        logger.info(f"Max dimensions: {model.get_max_dimensions()}")
        logger.info(f"Default dimensions: {model.get_default_dimensions()}")
        logger.info("Service is ready to accept requests")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "service": "Vietnamese Embedding Service",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "app.main:app", 
        host=host, 
        port=port, 
        reload=reload,
        log_level=log_level.lower()
    )