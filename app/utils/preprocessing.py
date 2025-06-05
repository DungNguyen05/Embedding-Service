from typing import List, Union

def clean_vietnamese_text(text: str) -> str:
    """Simple text cleaning."""
    if not text:
        return ""
    return text.strip()

def preprocess_texts(texts: Union[str, List[str]]) -> List[str]:
    """Simple preprocessing."""
    if isinstance(texts, str):
        texts = [texts]
    
    return [clean_vietnamese_text(text) for text in texts]

def truncate_text(text: str, max_length: int = 512) -> str:
    """Simple truncation."""
    if len(text) <= max_length:
        return text
    return text[:max_length].strip()