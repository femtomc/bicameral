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


# LanceDB implementation
class LanceDBVectorStore(VectorStore):
    """LanceDB-based vector store.
    
    Advantages:
    - Columnar storage (efficient)
    - Built-in vector indexing
    - SQL-like queries
    - Versioning support
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import lancedb
            import pyarrow as pa
            self.lancedb = lancedb
            self.pa = pa
            
            # Connect to LanceDB
            self.db = lancedb.connect(str(self.storage_path))
            
            # Define schema for embeddings table
            self.schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32())),
                pa.field("metadata", pa.string()),  # JSON string
                pa.field("timestamp", pa.string()),
                pa.field("type", pa.string()),
                pa.field("text", pa.string()),
            ])
            
            # Create or open table
            self.table_name = "embeddings"
            if self.table_name not in self.db.table_names():
                # Create empty table with schema
                self.table = self.db.create_table(
                    self.table_name,
                    schema=self.schema
                )
            else:
                self.table = self.db.open_table(self.table_name)
                
            logger.info(f"LanceDB store initialized at {storage_path}")
            
        except ImportError:
            logger.warning("LanceDB not installed, falling back to basic vector store")
            super().__init__(storage_path)
            
    async def add_embedding(self, 
                          embedding_id: str,
                          embedding: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None):
        """Add a new embedding with metadata."""
        if not hasattr(self, 'lancedb'):
            return await super().add_embedding(embedding_id, embedding, metadata)
            
        # Prepare data
        meta = metadata or {}
        meta['id'] = embedding_id
        timestamp = datetime.now().isoformat()
        
        # Create record
        data = [{
            "id": embedding_id,
            "vector": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "metadata": json.dumps(meta),
            "timestamp": timestamp,
            "type": meta.get('type', 'unknown'),
            "text": meta.get('text', ''),
        }]
        
        # Add to table
        self.table.add(data)
        logger.debug(f"Added embedding {embedding_id} to LanceDB")
        
    async def search_similar(self,
                           query_embedding: np.ndarray,
                           k: int = 5,
                           threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings using LanceDB's vector search."""
        if not hasattr(self, 'lancedb'):
            return await super().search_similar(query_embedding, k, threshold)
            
        # Convert to list if numpy array
        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Perform vector search
        results = (
            self.table.search(query_vec)
            .limit(k)
            .to_pandas()
        )
        
        # Process results
        output = []
        for _, row in results.iterrows():
            # Calculate cosine similarity from distance
            # LanceDB returns L2 distance by default
            distance = row.get('_distance', 0)
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            
            if similarity >= threshold:
                metadata = json.loads(row['metadata'])
                output.append((
                    row['id'],
                    float(similarity),
                    metadata
                ))
                
        return output
        
    async def get_embedding(self, embedding_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve embedding by ID."""
        if not hasattr(self, 'lancedb'):
            return await super().get_embedding(embedding_id)
            
        # Query by ID
        results = self.table.search().where(f"id = '{embedding_id}'").limit(1).to_pandas()
        
        if len(results) > 0:
            row = results.iloc[0]
            vector = np.array(row['vector'])
            metadata = json.loads(row['metadata'])
            return vector, metadata
            
        return None
        
    async def delete_embedding(self, embedding_id: str):
        """Delete an embedding."""
        if not hasattr(self, 'lancedb'):
            return await super().delete_embedding(embedding_id)
            
        # LanceDB doesn't support direct deletion yet
        # Mark as deleted in metadata instead
        result = await self.get_embedding(embedding_id)
        if result:
            vector, metadata = result
            metadata['deleted'] = True
            await self.add_embedding(embedding_id + "_deleted", vector, metadata)
            
    async def batch_add_embeddings(self,
                                 embeddings: List[Tuple[str, np.ndarray, Dict[str, Any]]]):
        """Add multiple embeddings efficiently."""
        if not hasattr(self, 'lancedb'):
            return await super().batch_add_embeddings(embeddings)
            
        # Prepare batch data
        data = []
        for embedding_id, embedding, metadata in embeddings:
            meta = metadata or {}
            meta['id'] = embedding_id
            timestamp = datetime.now().isoformat()
            
            data.append({
                "id": embedding_id,
                "vector": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "metadata": json.dumps(meta),
                "timestamp": timestamp,
                "type": meta.get('type', 'unknown'),
                "text": meta.get('text', ''),
            })
            
        # Batch add
        self.table.add(data)
        logger.info(f"Batch added {len(data)} embeddings to LanceDB")
        
    async def clear(self):
        """Clear all embeddings."""
        if not hasattr(self, 'lancedb'):
            return await super().clear()
            
        # Drop and recreate table
        self.db.drop_table(self.table_name)
        self.table = self.db.create_table(
            self.table_name,
            schema=self.schema
        )
        
    def statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        if not hasattr(self, 'lancedb'):
            return super().statistics()
            
        count = len(self.table.to_pandas())
        
        # Get storage size
        storage_size = 0
        for path in self.storage_path.rglob("*"):
            if path.is_file():
                storage_size += path.stat().st_size
                
        return {
            'count': count,
            'dimension': len(self.schema.field('vector').type.value_type) if count > 0 else 0,
            'storage_size_mb': storage_size / (1024 * 1024),
            'backend': 'lancedb'
        }


# ChromaDB implementation  
class ChromaDBVectorStore(VectorStore):
    """ChromaDB-based vector store.
    
    Advantages:
    - Purpose-built for embeddings
    - Multiple embedding functions
    - Built-in persistence
    - Good metadata filtering
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.chromadb = chromadb
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.storage_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection_name = "embeddings"
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Bicamrl embeddings storage"}
            )
            
            logger.info(f"ChromaDB store initialized at {storage_path}")
            
        except ImportError:
            logger.warning("ChromaDB not installed, falling back to basic vector store")
            super().__init__(storage_path)
            
    async def add_embedding(self, 
                          embedding_id: str,
                          embedding: np.ndarray,
                          metadata: Optional[Dict[str, Any]] = None):
        """Add a new embedding with metadata."""
        if not hasattr(self, 'chromadb'):
            return await super().add_embedding(embedding_id, embedding, metadata)
            
        # Prepare metadata
        meta = metadata or {}
        meta['timestamp'] = datetime.now().isoformat()
        
        # ChromaDB expects embeddings as lists
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Add to collection
        self.collection.add(
            embeddings=[embedding_list],
            ids=[embedding_id],
            metadatas=[meta],
            documents=[meta.get('text', '')]  # ChromaDB requires documents
        )
        
        logger.debug(f"Added embedding {embedding_id} to ChromaDB")
        
    async def search_similar(self,
                           query_embedding: np.ndarray,
                           k: int = 5,
                           threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings using ChromaDB."""
        if not hasattr(self, 'chromadb'):
            return await super().search_similar(query_embedding, k, threshold)
            
        # Convert to list
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=k
        )
        
        # Process results
        output = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                embedding_id = results['ids'][0][i]
                # ChromaDB returns distances, convert to similarity
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)
                
                if similarity >= threshold:
                    metadata = results['metadatas'][0][i]
                    output.append((
                        embedding_id,
                        float(similarity),
                        metadata
                    ))
                    
        return output
        
    async def get_embedding(self, embedding_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve embedding by ID."""
        if not hasattr(self, 'chromadb'):
            return await super().get_embedding(embedding_id)
            
        # Get by ID
        results = self.collection.get(
            ids=[embedding_id],
            include=['embeddings', 'metadatas']
        )
        
        if results['ids'] and len(results['ids']) > 0:
            embedding = np.array(results['embeddings'][0])
            metadata = results['metadatas'][0]
            return embedding, metadata
            
        return None
        
    async def delete_embedding(self, embedding_id: str):
        """Delete an embedding."""
        if not hasattr(self, 'chromadb'):
            return await super().delete_embedding(embedding_id)
            
        # ChromaDB supports direct deletion
        self.collection.delete(ids=[embedding_id])
        logger.debug(f"Deleted embedding {embedding_id} from ChromaDB")
        
    async def batch_add_embeddings(self,
                                 embeddings: List[Tuple[str, np.ndarray, Dict[str, Any]]]):
        """Add multiple embeddings efficiently."""
        if not hasattr(self, 'chromadb'):
            return await super().batch_add_embeddings(embeddings)
            
        # Prepare batch data
        ids = []
        embedding_lists = []
        metadatas = []
        documents = []
        
        for embedding_id, embedding, metadata in embeddings:
            meta = metadata or {}
            meta['timestamp'] = datetime.now().isoformat()
            
            ids.append(embedding_id)
            embedding_lists.append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            metadatas.append(meta)
            documents.append(meta.get('text', ''))
            
        # Batch add
        self.collection.add(
            embeddings=embedding_lists,
            ids=ids,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Batch added {len(ids)} embeddings to ChromaDB")
        
    async def clear(self):
        """Clear all embeddings."""
        if not hasattr(self, 'chromadb'):
            return await super().clear()
            
        # Delete collection and recreate
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Bicamrl embeddings storage"}
        )
        
    def statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        if not hasattr(self, 'chromadb'):
            return super().statistics()
            
        count = self.collection.count()
        
        # Get storage size
        storage_size = 0
        for path in self.storage_path.rglob("*"):
            if path.is_file():
                storage_size += path.stat().st_size
                
        # Get a sample to determine dimension
        dimension = 0
        if count > 0:
            sample = self.collection.get(limit=1, include=['embeddings'])
            if sample['embeddings']:
                dimension = len(sample['embeddings'][0])
                
        return {
            'count': count,
            'dimension': dimension,
            'storage_size_mb': storage_size / (1024 * 1024),
            'backend': 'chromadb'
        }