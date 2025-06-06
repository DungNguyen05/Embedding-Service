import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import os

class VietnameseEmbeddingModel:
    """Simple embedding model wrapper with flexible dimensions."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.max_dimensions = self.model.get_sentence_embedding_dimension()
            self.default_dimensions = int(os.getenv("DEFAULT_EMBEDDING_DIMENSIONS", str(self.max_dimensions)))
        except Exception as e:
            # Fallback to basic model
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.max_dimensions = self.model.get_sentence_embedding_dimension()
            self.default_dimensions = int(os.getenv("DEFAULT_EMBEDDING_DIMENSIONS", str(self.max_dimensions)))
    
    def encode(self, texts: Union[str, List[str]], dimensions: Optional[int] = None, 
               normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """Encode texts into embeddings with flexible dimensions."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Apply dimension truncation if requested
        target_dimensions = dimensions or self.default_dimensions
        if target_dimensions and target_dimensions < embeddings.shape[1]:
            embeddings = embeddings[:, :target_dimensions]
        
        return embeddings
    
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