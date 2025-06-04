import re
import unicodedata
from typing import List, Union

def clean_vietnamese_text(text: str) -> str:
    """
    Clean and normalize Vietnamese text for better embedding quality.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize Unicode characters (important for Vietnamese diacritics)
    text = unicodedata.normalize('NFC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove special characters that might interfere with embedding quality
    # Keep Vietnamese characters, alphanumeric, and basic punctuation
    text = re.sub(r'[^\w\s\.,!?;:()\[\]"\'-àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]', ' ', text)
    
    # Clean up any double spaces created by the above operation
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_texts(texts: Union[str, List[str]]) -> List[str]:
    """
    Preprocess a single text or list of texts for embedding.
    
    Args:
        texts: Single text string or list of text strings
        
    Returns:
        List of preprocessed texts
    """
    if isinstance(texts, str):
        texts = [texts]
    
    processed_texts = []
    for text in texts:
        cleaned_text = clean_vietnamese_text(text)
        
        # Skip empty texts after cleaning
        if cleaned_text:
            processed_texts.append(cleaned_text)
        else:
            # Add a placeholder for empty texts to maintain indexing
            processed_texts.append(" ")
    
    return processed_texts

def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length while trying to preserve word boundaries.
    
    Args:
        text: Input text
        max_length: Maximum character length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to cut at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can cut at a word boundary near the end
        return truncated[:last_space].strip()
    else:
        return truncated.strip()