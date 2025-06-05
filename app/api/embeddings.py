from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import time
import logging
import traceback
import gc
import torch
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
        logger.info(f"=== Starting embedding request ===")
        logger.info(f"Input type: {type(request.input)}")
        logger.info(f"Input content: {request.input}")
        
        # Get the embedding model
        logger.info("Getting embedding model...")
        model = get_embedding_model()
        logger.info(f"Model retrieved: {model.get_model_name()}")
        
        # Prepare input texts
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        logger.info(f"Number of texts to process: {len(texts)}")
        
        # Preprocess texts with error handling
        try:
            logger.info("Starting text preprocessing...")
            processed_texts = preprocess_texts(texts)
            logger.info(f"Preprocessing successful. Results: {processed_texts}")
        except Exception as prep_error:
            logger.error(f"Preprocessing failed: {prep_error}")
            logger.error(f"Preprocessing traceback: {traceback.format_exc()}")
            # Use original texts if preprocessing fails
            processed_texts = texts
        
        # Truncate texts if they're too long
        try:
            logger.info("Starting text truncation...")
            truncated_texts = [truncate_text(text, max_length=256) for text in processed_texts]  # Reduced max_length
            logger.info(f"Truncation successful. Results: {truncated_texts}")
        except Exception as trunc_error:
            logger.error(f"Truncation failed: {trunc_error}")
            logger.error(f"Truncation traceback: {traceback.format_exc()}")
            # Use processed texts if truncation fails
            truncated_texts = processed_texts
        
        logger.info(f"About to generate embeddings for: {truncated_texts}")
        
        # Force garbage collection before heavy operation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate embeddings with comprehensive error handling
        try:
            logger.info("=== Starting embedding generation ===")
            
            # Generate embeddings with correct parameters
            embeddings = model.encode(
                truncated_texts, 
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            logger.info(f"Embedding generation successful!")
            logger.info(f"Embeddings shape: {embeddings.shape}")
            logger.info(f"Embeddings type: {type(embeddings)}")
            
        except Exception as embed_error:
            logger.error(f"=== EMBEDDING GENERATION FAILED ===")
            logger.error(f"Error type: {type(embed_error)}")
            logger.error(f"Error message: {str(embed_error)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try to get more system info
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                logger.error(f"Available memory: {memory_info.available / 1024 / 1024:.2f} MB")
                logger.error(f"Memory percent used: {memory_info.percent}%")
            except:
                logger.error("Could not get memory info")
            
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(embed_error)}")
        
        # Prepare response data
        try:
            logger.info("Preparing response data...")
            embedding_data = []
            for i, embedding in enumerate(embeddings):
                embedding_data.append(EmbeddingData(
                    embedding=embedding.tolist(),
                    index=i
                ))
            logger.info(f"Response data prepared for {len(embedding_data)} embeddings")
        except Exception as resp_error:
            logger.error(f"Response preparation failed: {resp_error}")
            logger.error(f"Response traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Response preparation failed: {str(resp_error)}")
        
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
        
        logger.info(f"=== Request completed successfully ===")
        logger.info(f"Generated {len(embedding_data)} embeddings")
        
        # Clean up
        gc.collect()
        
        return response
        
    except HTTPException:
        logger.error("HTTPException occurred, re-raising")
        raise
    except Exception as e:
        logger.error(f"=== UNEXPECTED ERROR ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Clean up on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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