"""LLM-based embeddings for vector storage."""

import asyncio
import aiohttp
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LLMEmbeddings:
    """Generate embeddings using LLM providers (OpenAI-compatible APIs)."""
    
    def __init__(self, base_url: str, api_key: str = "not-needed", model: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.dimension = None  # Will be set after first embedding
        
    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else None
        
    async def embed_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []
            
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "input": texts,
            "model": self.model or "text-embedding-nomic-embed-text-v1.5"  # Common LM Studio embedding model
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract embeddings
                        embeddings = []
                        for item in result.get("data", []):
                            embedding = item.get("embedding")
                            if embedding:
                                arr = np.array(embedding, dtype=np.float32)
                                embeddings.append(arr)
                                
                                # Set dimension on first successful embedding
                                if self.dimension is None:
                                    self.dimension = len(embedding)
                            else:
                                embeddings.append(None)
                                
                        return embeddings
                    else:
                        error_text = await response.text()
                        logger.error(f"Embedding API error {response.status}: {error_text}")
                        return [None] * len(texts)
                        
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return [None] * len(texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension or 768  # Default dimension


class LLMEmbeddingAdapter:
    """Adapter to make LLM embeddings work with the hybrid store."""
    
    def __init__(self, llm_embeddings: LLMEmbeddings):
        self.llm_embeddings = llm_embeddings
        self._dimension = None
        
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings (compatible with sentence-transformers interface)."""
        embeddings = await self.llm_embeddings.embed_texts(texts)
        
        # Handle None values by using zero vectors
        dimension = self.llm_embeddings.get_dimension()
        result = []
        
        for emb in embeddings:
            if emb is not None:
                result.append(emb)
            else:
                # Use zero vector for failed embeddings
                result.append(np.zeros(dimension, dtype=np.float32))
                
        return np.array(result)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._dimension = self.llm_embeddings.get_dimension()
        return self._dimension