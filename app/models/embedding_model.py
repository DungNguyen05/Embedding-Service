import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import os
import logging

logger = logging.getLogger(__name__)

class VietnameseEmbeddingModel:
    """Simple embedding model wrapper with flexible dimensions."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"Loading model: {self.model_name} on device: {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.max_dimensions = self.model.get_sentence_embedding_dimension()
            
            # Ensure default dimensions don't exceed model's maximum
            env_default = int(os.getenv("DEFAULT_EMBEDDING_DIMENSIONS", str(self.max_dimensions)))
            self.default_dimensions = min(env_default, self.max_dimensions)
            
            if env_default > self.max_dimensions:
                logger.warning(f"DEFAULT_EMBEDDING_DIMENSIONS ({env_default}) exceeds model max ({self.max_dimensions}). Using {self.default_dimensions}")
            
            logger.info(f"Model loaded successfully. Max dimensions: {self.max_dimensions}")
        except Exception as e:
            logger.warning(f"Failed to load primary model {self.model_name}: {e}")
            logger.info("Falling back to basic model")
            # Fallback to basic model
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.max_dimensions = self.model.get_sentence_embedding_dimension()
            
            # Ensure default dimensions don't exceed model's maximum
            env_default = int(os.getenv("DEFAULT_EMBEDDING_DIMENSIONS", str(self.max_dimensions)))
            self.default_dimensions = min(env_default, self.max_dimensions)
            
            if env_default > self.max_dimensions:
                logger.warning(f"DEFAULT_EMBEDDING_DIMENSIONS ({env_default}) exceeds model max ({self.max_dimensions}). Using {self.default_dimensions}")
            
            logger.info(f"Fallback model loaded. Max dimensions: {self.max_dimensions}")
    
    def encode(self, texts: Union[str, List[str]], dimensions: Optional[int] = None, 
               normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """Encode texts into embeddings with flexible dimensions."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean empty texts
        processed_texts = []
        for text in texts:
            clean_text = str(text).strip()
            if not clean_text:
                clean_text = " "  # Use space for empty texts
            processed_texts.append(clean_text)
        
        try:
            embeddings = self.model.encode(
                processed_texts, 
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Apply dimension truncation if requested
            target_dimensions = dimensions or self.default_dimensions
            if target_dimensions and target_dimensions < embeddings.shape[1]:
                embeddings = embeddings[:, :target_dimensions]
            
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def get_max_dimensions(self) -> int:
        """Get the maximum dimensions supported by the model."""
        return self.max_dimensions
    
    def get_default_dimensions(self) -> int:
        """Get the default dimensions from configuration."""
        return self.default_dimensions
    
    def get_model_name(self) -> str:
        return self.model_name

# Global model instance
_model_instance = None

def get_embedding_model() -> VietnameseEmbeddingModel:
    """Get or create the global embedding model instance."""
    global _model_instance
    if _model_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME")
        device = os.getenv("DEVICE", "cpu")
        _model_instance = VietnameseEmbeddingModel(model_name=model_name, device=device)
    return _model_instance