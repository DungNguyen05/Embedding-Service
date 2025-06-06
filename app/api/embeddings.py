from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import time
from app.models.embedding_model import get_embedding_model

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
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== EMBEDDING REQUEST STARTED ===")
        logger.info(f"Request: {request}")
        
        model = get_embedding_model()
        logger.info("Model retrieved successfully")
        
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
        
        logger.info(f"Processing {len(texts)} text(s): {texts}")
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Simple preprocessing - just strip and truncate
        processed_texts = []
        for text in texts:
            clean_text = text.strip()[:512]  # Simple truncation
            processed_texts.append(clean_text if clean_text else " ")
        
        logger.info(f"Processed texts: {processed_texts}")
        
        # Generate embeddings with requested dimensions
        logger.info("Generating embeddings...")
        embeddings = model.encode(
            processed_texts, 
            dimensions=request.dimensions,
            normalize_embeddings=True
        )
        logger.info(f"Embeddings generated with shape: {embeddings.shape}")
        
        # Prepare response
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(EmbeddingData(
                embedding=embedding.tolist(),
                index=i
            ))
        
        # Simple token estimation
        total_chars = sum(len(text) for text in processed_texts)
        estimated_tokens = max(1, total_chars // 4)
        
        response = EmbeddingResponse(
            data=embedding_data,
            model="text-embedding-ada-002",
            usage=EmbeddingUsage(
                prompt_tokens=estimated_tokens,
                total_tokens=estimated_tokens
            )
        )
        
        logger.info("=== EMBEDDING REQUEST COMPLETED ===")
        return response
        
    except Exception as e:
        logger.error(f"=== EMBEDDING REQUEST FAILED: {str(e)} ===")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models - OpenAI compatible."""
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

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        model = get_embedding_model()
        return {
            "status": "healthy",
            "model": model.get_model_name(),
            "max_dimensions": model.get_max_dimensions(),
            "default_dimensions": model.get_default_dimensions()
        }
    except Exception:
        raise HTTPException(status_code=503, detail="Service unhealthy")