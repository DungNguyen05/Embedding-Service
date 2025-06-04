from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from dotenv import load_dotenv
from app.api.embeddings import router as embeddings_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vietnamese Embedding Service",
    description="OpenAI-compatible embedding API optimized for Vietnamese language",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(embeddings_router, tags=["embeddings"])

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Vietnamese Embedding Service...")
    
    # Pre-load the model to avoid cold start delays
    try:
        from app.models.embedding_model import get_embedding_model
        model = get_embedding_model()
        logger.info(f"Model pre-loaded: {model.get_model_name()}")
        logger.info(f"Model dimensions: {model.get_dimensions()}")
        logger.info("Service startup complete!")
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down Vietnamese Embedding Service...")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Vietnamese Embedding Service",
        "version": "1.0.0",
        "description": "OpenAI-compatible embedding API optimized for Vietnamese language",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/info")
async def service_info():
    """Get detailed service information."""
    try:
        from app.models.embedding_model import get_embedding_model
        model = get_embedding_model()
        
        return {
            "service": "Vietnamese Embedding Service",
            "model": {
                "name": model.get_model_name(),
                "dimensions": model.get_dimensions(),
                "device": model.device
            },
            "api_compatibility": "OpenAI v1",
            "supported_languages": ["Vietnamese", "English", "Multilingual"],
            "features": [
                "Vietnamese text preprocessing",
                "Unicode normalization",
                "Batch embedding generation",
                "OpenAI API compatibility"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting service info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )