import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import os
import logging

logger = logging.getLogger(__name__)

class VietnameseEmbeddingModel:
    """Vietnamese embedding model optimized for RAG applications."""
    
    def __init__(self, model_name: str = None, device: str = None, cache_folder: str = None):
        self.model_name = model_name or "keepitreal/vietnamese-sbert"
        
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Set custom cache directory
        self.cache_folder = cache_folder or os.getenv("MODEL_CACHE_DIR", "./models_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_folder, exist_ok=True)
        
        logger.info(f"Loading Vietnamese model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Cache directory: {self.cache_folder}")
        
        try:
            # Load model with custom cache folder
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                cache_folder=self.cache_folder
            )
            
            # Get maximum dimensions for best RAG performance
            self.max_dimensions = self.model.get_sentence_embedding_dimension()
            
            # Use maximum dimensions by default for best RAG quality
            env_default = int(os.getenv("DEFAULT_EMBEDDING_DIMENSIONS", str(self.max_dimensions)))
            self.default_dimensions = min(env_default, self.max_dimensions)
            
            # Warn if not using maximum dimensions
            if self.default_dimensions < self.max_dimensions:
                logger.warning(f"Using {self.default_dimensions} dimensions instead of maximum {self.max_dimensions}")
                logger.warning("For best RAG performance, consider using maximum dimensions")
            
            logger.info(f"Vietnamese model loaded successfully!")
            logger.info(f"Model dimensions: {self.max_dimensions}")
            logger.info(f"Using dimensions: {self.default_dimensions}")
            
        except Exception as e:
            logger.error(f"Failed to load Vietnamese model {self.model_name}: {e}")
            raise Exception(f"Could not load Vietnamese embedding model: {e}")
    
    def encode(self, texts: Union[str, List[str]], dimensions: Optional[int] = None, 
               normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """Encode Vietnamese texts into embeddings optimized for RAG."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean and prepare Vietnamese texts
        processed_texts = []
        for text in texts:
            clean_text = str(text).strip()
            if not clean_text:
                clean_text = " "  # Use space for empty texts
            processed_texts.append(clean_text)
        
        try:
            # Generate embeddings with Vietnamese-optimized model
            embeddings = self.model.encode(
                processed_texts, 
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32  # Optimized batch size for Vietnamese model
            )
            
            # Apply dimension truncation if requested (not recommended for RAG)
            target_dimensions = dimensions or self.default_dimensions
            if target_dimensions and target_dimensions < embeddings.shape[1]:
                logger.warning(f"Truncating embeddings from {embeddings.shape[1]} to {target_dimensions} dimensions")
                logger.warning("This may reduce RAG performance quality")
                embeddings = embeddings[:, :target_dimensions]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding Vietnamese texts: {e}")
            raise
    
    def get_max_dimensions(self) -> int:
        """Get the maximum dimensions supported by the Vietnamese model."""
        return self.max_dimensions
    
    def get_default_dimensions(self) -> int:
        """Get the configured dimensions."""
        return self.default_dimensions
    
    def get_model_name(self) -> str:
        """Get the Vietnamese model name."""
        return self.model_name
    
    def get_cache_folder(self) -> str:
        """Get the model cache directory path."""
        return self.cache_folder

# Global model instance
_model_instance = None

def get_embedding_model() -> VietnameseEmbeddingModel:
    """Get or create the global Vietnamese embedding model instance."""
    global _model_instance
    if _model_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "keepitreal/vietnamese-sbert")
        device = os.getenv("DEVICE", "auto")
        cache_folder = os.getenv("MODEL_CACHE_DIR", "./models_cache")
        
        _model_instance = VietnameseEmbeddingModel(
            model_name=model_name, 
            device=device,
            cache_folder=cache_folder
        )
    return _model_instance