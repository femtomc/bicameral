"""Vector storage for embeddings and similarity search."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pickle
from datetime import datetime

from ..utils.logging_config import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """Simple vector storage with similarity search.
    
    This is a basic implementation that can be replaced with
    LanceDB, ChromaDB, or other vector databases as needed.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Paths for different data
        self.embeddings_path = self.storage_path / "embeddings.npy"
        self.metadata_path = self.storage_path / "metadata.json"
        self.index_path = self.storage_path / "index.pkl"
        
        # In-memory data
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_index: Dict[str, int] = {}
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load existing embeddings and metadata."""
        try:
            if self.embeddings_path.exists():
                self.embeddings = np.load(self.embeddings_path)
                logger.info(f"Loaded {len(self.embeddings)} embeddings")
            
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            if self.index_path.exists():
                with open(self.index_path, 'rb') as f:
                    self.id_to_index = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading vector data: {e}")
            self.embeddings = None
            self.metadata = []
            self.id_to_index = {}
    
    def _save_data(self):
        """Persist embeddings and metadata to disk."""
        try:
            if self.embeddings is not None:
                np.save(self.embeddings_path, self.embeddings)
            
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.id_to_index, f)
        except Exception as e:
            logger.error(f"Error saving vector data: {e}")
    
    async def add_embedding(self, 
                          embedding_id: str,
                          embedding: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None):
        """Add a new embedding with metadata."""
        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Initialize or append to embeddings
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
            index = 0
        else:
            # Check dimension compatibility
            if embedding.shape[0] != self.embeddings.shape[1]:
                raise ValueError(f"Embedding dimension {embedding.shape[0]} doesn't match "
                               f"existing dimension {self.embeddings.shape[1]}")
            
            index = len(self.embeddings)
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        # Store metadata
        meta = metadata or {}
        meta['id'] = embedding_id
        meta['index'] = index
        meta['timestamp'] = datetime.now().isoformat()
        self.metadata.append(meta)
        
        # Update index
        self.id_to_index[embedding_id] = index
        
        # Save periodically (every 10 additions)
        if len(self.metadata) % 10 == 0:
            self._save_data()
    
    async def search_similar(self,
                           query_embedding: np.ndarray,
                           k: int = 5,
                           threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings using cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Ensure query is numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize all embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / norms
        
        # Compute cosine similarities
        similarities = np.dot(normalized_embeddings, query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= threshold:
                metadata = self.metadata[idx]
                embedding_id = metadata['id']
                results.append((embedding_id, similarity, metadata))
        
        return results
    
    async def get_embedding(self, embedding_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve embedding by ID."""
        if embedding_id in self.id_to_index:
            index = self.id_to_index[embedding_id]
            return self.embeddings[index], self.metadata[index]
        return None
    
    async def delete_embedding(self, embedding_id: str):
        """Delete an embedding (marks as deleted, doesn't remove from arrays)."""
        if embedding_id in self.id_to_index:
            index = self.id_to_index[embedding_id]
            self.metadata[index]['deleted'] = True
            del self.id_to_index[embedding_id]
            self._save_data()
    
    async def batch_add_embeddings(self,
                                 embeddings: List[Tuple[str, np.ndarray, Dict[str, Any]]]):
        """Add multiple embeddings efficiently."""
        for embedding_id, embedding, metadata in embeddings:
            await self.add_embedding(embedding_id, embedding, metadata)
        
        # Force save after batch
        self._save_data()
    
    async def clear(self):
        """Clear all embeddings."""
        self.embeddings = None
        self.metadata = []
        self.id_to_index = {}
        
        # Remove files
        for path in [self.embeddings_path, self.metadata_path, self.index_path]:
            if path.exists():
                path.unlink()
    
    def statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        if self.embeddings is None:
            return {
                'count': 0,
                'dimension': 0,
                'storage_size_mb': 0
            }
        
        storage_size = 0
        for path in [self.embeddings_path, self.metadata_path, self.index_path]:
            if path.exists():
                storage_size += path.stat().st_size
        
        return {
            'count': len(self.embeddings),
            'dimension': self.embeddings.shape[1],
            'storage_size_mb': storage_size / (1024 * 1024),
            'deleted_count': sum(1 for m in self.metadata if m.get('deleted', False))
        }


# Future: LanceDB implementation
class LanceDBVectorStore(VectorStore):
    """LanceDB-based vector store (to be implemented).
    
    Advantages:
    - Columnar storage (efficient)
    - Built-in vector indexing
    - SQL-like queries
    - Versioning support
    """
    
    def __init__(self, storage_path: Path):
        super().__init__(storage_path)
        # TODO: Implement with lancedb when ready
        logger.info("LanceDB store initialized (using basic store for now)")


# Future: ChromaDB implementation  
class ChromaDBVectorStore(VectorStore):
    """ChromaDB-based vector store (to be implemented).
    
    Advantages:
    - Purpose-built for embeddings
    - Multiple embedding functions
    - Built-in persistence
    - Good metadata filtering
    """
    
    def __init__(self, storage_path: Path):
        super().__init__(storage_path)
        # TODO: Implement with chromadb when ready
        logger.info("ChromaDB store initialized (using basic store for now)")