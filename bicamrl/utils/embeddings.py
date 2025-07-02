"""Utilities for working with embeddings and text processing."""

import re
from typing import List, Optional
import numpy as np


def preprocess_text_for_embedding(text: str) -> str:
    """Preprocess text before embedding generation."""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove code blocks markers but keep content
    text = re.sub(r'```\w*\n?', ' ', text)
    
    # Keep alphanumeric, spaces, and common punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
    
    # Normalize spaces again
    text = ' '.join(text.split())
    
    # Truncate to reasonable length (most models have token limits)
    max_length = 512
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def combine_embeddings(embeddings: List[np.ndarray], method: str = 'mean') -> np.ndarray:
    """Combine multiple embeddings into one."""
    if not embeddings:
        raise ValueError("No embeddings to combine")
    
    embeddings_array = np.array(embeddings)
    
    if method == 'mean':
        return np.mean(embeddings_array, axis=0)
    elif method == 'max':
        return np.max(embeddings_array, axis=0)
    elif method == 'weighted':
        # Give more weight to earlier chunks
        weights = np.array([1.0 / (i + 1) for i in range(len(embeddings))])
        weights = weights / weights.sum()
        return np.average(embeddings_array, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown combination method: {method}")


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Normalize
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(embedding1, embedding2) / (norm1 * norm2)


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate Euclidean distance between two embeddings."""
    return np.linalg.norm(embedding1 - embedding2)


class EmbeddingCache:
    """Simple cache for embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if text in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(text)
            self.access_order.append(text)
            return self.cache[text]
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        if text in self.cache:
            # Update and move to end
            self.access_order.remove(text)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_text = self.access_order.pop(0)
            del self.cache[lru_text]
        
        self.cache[text] = embedding
        self.access_order.append(text)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """Extract key phrases from text (simple implementation)."""
    # Remove common words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'then', 'once', 'is', 'are', 'was', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what',
        'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'some', 'any', 'few', 'many', 'much', 'most', 'other', 'another', 'such',
        'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Count frequencies
    from collections import Counter
    word_freq = Counter(keywords)
    
    # Get top words
    top_words = [word for word, _ in word_freq.most_common(max_phrases * 2)]
    
    # Try to find bigrams
    phrases = []
    for i in range(len(words) - 1):
        if words[i] not in stop_words and words[i+1] not in stop_words:
            bigram = f"{words[i]} {words[i+1]}"
            if len(phrases) < max_phrases:
                phrases.append(bigram)
    
    # Add top single words if needed
    for word in top_words:
        if len(phrases) < max_phrases:
            phrases.append(word)
    
    return phrases[:max_phrases]