from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import time
import logging
from app.models.embedding_model import get_embedding_model
from app.utils.preprocessing import preprocess_texts, truncate_text

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models matching OpenAI API format
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default="text-embedding-ada-002", description="Model name (ignored, using Vietnamese model)")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions (ignored, using model default)")
    user: Optional[str] = Field(default=None, description="User identifier")

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

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for the given input text(s).
    Compatible with OpenAI's embeddings API format.
    """
    try:
        # Get the embedding model
        model = get_embedding_model()
        
        # Prepare input texts
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Preprocess texts
        processed_texts = preprocess_texts(texts)
        
        # Truncate texts if they're too long (most models have token limits)
        truncated_texts = [truncate_text(text, max_length=2048) for text in processed_texts]
        
        logger.info(f"Processing {len(truncated_texts)} texts for embeddings")
        
        # Generate embeddings
        embeddings = model.encode(truncated_texts, normalize_embeddings=True)
        
        # Prepare response data
        embedding_data = []
        for i, embedding in enumerate(embeddings):
            embedding_data.append(EmbeddingData(
                embedding=embedding.tolist(),
                index=i
            ))
        
        # Calculate usage (rough estimate)
        total_chars = sum(len(text) for text in truncated_texts)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
        response = EmbeddingResponse(
            data=embedding_data,
            model=model.get_model_name(),
            usage=EmbeddingUsage(
                prompt_tokens=estimated_tokens,
                total_tokens=estimated_tokens
            )
        )
        
        logger.info(f"Successfully generated {len(embedding_data)} embeddings")
        return response
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """
    List available models.
    Compatible with OpenAI's models API format.
    """
    try:
        model = get_embedding_model()
        
        models = [
            ModelInfo(
                id="text-embedding-ada-002",  # OpenAI compatible name
                created=int(time.time()),
            ),
            ModelInfo(
                id="vietnamese-embedding",  # Our actual model name
                created=int(time.time()),
            ),
            ModelInfo(
                id=model.get_model_name(),  # Real model name
                created=int(time.time()),
            )
        ]
        
        return ModelListResponse(data=models)
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        model = get_embedding_model()
        return {
            "status": "healthy",
            "model": model.get_model_name(),
            "dimensions": model.get_dimensions()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    Get information about a specific model.
    Compatible with OpenAI's model retrieval API.
    """
    try:
        model = get_embedding_model()
        
        return ModelInfo(
            id=model_id,
            created=int(time.time()),
        )
        
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")