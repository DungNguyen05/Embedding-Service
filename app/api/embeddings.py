from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import time
import logging
from app.models.embedding_model import get_embedding_model

logger = logging.getLogger(__name__)
router = APIRouter()

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default="text-embedding-ada-002")
    encoding_format: Optional[str] = Field(default="float")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions for the embeddings")

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vietnamese-embedding-service"
    max_dimensions: Optional[int] = None
    default_dimensions: Optional[int] = None

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings - OpenAI compatible."""
    try:
        logger.debug(f"Embedding request received for {len(request.input) if isinstance(request.input, list) else 1} text(s)")
        
        model = get_embedding_model()
        
        # Validate dimensions if provided
        if request.dimensions:
            max_dims = model.get_max_dimensions()
            if request.dimensions > max_dims:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Requested dimensions ({request.dimensions}) exceed model maximum ({max_dims})"
                )
            if request.dimensions < 1:
                raise HTTPException(
                    status_code=400, 
                    detail="Dimensions must be a positive integer"
                )
        
        # Prepare texts
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Generate embeddings with requested dimensions
        embeddings = model.encode(
            texts, 
            dimensions=request.dimensions,
            normalize_embeddings=True
        )
        
        # Prepare response
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(EmbeddingData(
                embedding=embedding.tolist(),
                index=i
            ))
        
        # Simple token estimation
        total_chars = sum(len(str(text)) for text in texts)
        estimated_tokens = max(1, total_chars // 4)
        
        response = EmbeddingResponse(
            data=embedding_data,
            model="text-embedding-ada-002",
            usage=EmbeddingUsage(
                prompt_tokens=estimated_tokens,
                total_tokens=estimated_tokens
            )
        )
        
        logger.debug(f"Successfully generated {len(embedding_data)} embeddings")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models - OpenAI compatible."""
    try:
        model = get_embedding_model()
        models = [
            ModelInfo(
                id="text-embedding-ada-002",
                created=int(time.time()),
                max_dimensions=model.get_max_dimensions(),
                default_dimensions=model.get_default_dimensions()
            )
        ]
        return ModelListResponse(data=models)
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        model = get_embedding_model()
        return {
            "status": "healthy",
            "model": model.get_model_name(),
            "max_dimensions": model.get_max_dimensions(),
            "default_dimensions": model.get_default_dimensions(),
            "device": model.device
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
    

@router.post("/v1/embeddings-test")
async def test_embeddings_simple(request: dict):
    """Simple test endpoint for debugging."""
    try:
        logger.info(f"Test request received: {request}")
        return {"status": "ok", "received": request}
    except Exception as e:
        logger.error(f"Test endpoint failed: {e}")
        return {"error": str(e)}