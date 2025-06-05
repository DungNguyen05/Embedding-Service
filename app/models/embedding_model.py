import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
import os

logger = logging.getLogger(__name__)

class VietnameseEmbeddingModel:
    """
    Vietnamese-optimized embedding model using multilingual sentence transformers.
    Uses models that perform well on Vietnamese text.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the model to use. Defaults to best Vietnamese model.
            device: Device to run the model on ('cuda', 'cpu', or 'auto')
        """
        # Best multilingual models for Vietnamese (ordered by performance)
        self.vietnamese_models = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Reliable and lightweight
            "sentence-transformers/LaBSE",  # Language-agnostic BERT (good for Vietnamese)
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # Good multilingual performance
            "sentence-transformers/all-MiniLM-L6-v2",  # Fallback option
        ]
        
        # Use provided model or default to best Vietnamese model
        self.model_name = model_name or self.vietnamese_models[0]
        
        # Auto-detect device if not specified
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        logger.info(f"Loading embedding model: {self.model_name} on device: {self.device}")
        
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get model dimensions
            self.dimensions = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Dimensions: {self.dimensions}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Try fallback models in order
            for fallback_model in self.vietnamese_models:
                if fallback_model != self.model_name:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        self.model_name = fallback_model
                        self.model = SentenceTransformer(self.model_name, device=self.device)
                        self.dimensions = self.model.get_sentence_embedding_dimension()
                        logger.info(f"Fallback model {fallback_model} loaded successfully. Dimensions: {self.dimensions}")
                        break
                    except Exception as fallback_error:
                        logger.error(f"Fallback model {fallback_model} also failed: {fallback_error}")
                        continue
            else:
                # If all models fail, raise the original error
                raise Exception(f"All embedding models failed to load. Last error: {e}")
    
    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text string or list of text strings
            normalize_embeddings: Whether to normalize embeddings to unit length
            **kwargs: Additional arguments (filtered to valid ones)
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Filter kwargs to only include valid parameters for sentence-transformers
            valid_kwargs = {}
            allowed_params = {
                'batch_size', 'show_progress_bar', 'output_value', 
                'convert_to_numpy', 'convert_to_tensor', 'device',
                'normalize_embeddings'
            }
            
            for key, value in kwargs.items():
                if key in allowed_params:
                    valid_kwargs[key] = value
            
            # Set sensible defaults
            encode_params = {
                'normalize_embeddings': normalize_embeddings,
                'convert_to_numpy': True,
                'show_progress_bar': False,
                **valid_kwargs
            }
            
            logger.info(f"Encoding {len(texts)} texts with params: {encode_params}")
            
            # Generate embeddings
            embeddings = self.model.encode(texts, **encode_params)
            
            logger.info(f"Successfully encoded {len(texts)} texts. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            logger.error(f"Texts: {texts}")
            logger.error(f"Parameters: {encode_params}")
            raise
    
    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.dimensions
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.model_name

# Global model instance
_model_instance = None

def get_embedding_model() -> VietnameseEmbeddingModel:
    """Get or create the global embedding model instance."""
    global _model_instance
    if _model_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME")
        device = os.getenv("DEVICE", "auto")
        _model_instance = VietnameseEmbeddingModel(model_name=model_name, device=device)
    return _model_instance