"""
FAISS Vector Store Implementation - Local vector similarity search
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Local FAISS vector store for vector similarity search"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FAISS vector store
        
        Args:
            config: Configuration dictionary with:
                - index_path: Path to save/load the index
                - index_type: Type of index (Flat, IVF, HNSW)
                - nlist: Number of clusters (for IVF)
        """
        self.index_path = config["index_path"]
        self.index_type = config.get("index_type", "Flat")
        self.nlist = config.get("nlist", 100)
        
        self.index = None
        self.dimension = None
        self.metadata_path = self.index_path.replace(".faiss", "_metadata.json")
        self.metadata = {}
        self.is_trained = False
        
        logger.info(f"🏭 Initializing FAISS vector store ({self.index_type})")
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one"""
        try:
            if os.path.exists(self.index_path):
                logger.info("📂 Loading existing FAISS index...")
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                self._load_metadata()
                logger.info(f"✅ FAISS index loaded ({len(self.metadata)} vectors)")
                self.is_trained = True
            else:
                logger.info("🆕 Creating new FAISS index...")
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        except Exception as e:
            logger.error(f"❌ Error loading FAISS index: {e}")
            raise
    
    def _load_metadata(self):
        """Load metadata from file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️  Could not load metadata: {e}")
                self.metadata = {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️  Could not save metadata: {e}")
    
    def add_vectors(self, vectors: List[np.ndarray], metadata_list: List[Dict[str, Any]]) -> List[int]:
        """Add vectors to the index
        
        Args:
            vectors: List of numpy arrays (embeddings)
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of vector IDs
        """
        if not vectors:
            logger.warning("⚠️  No vectors to add")
            return []
        
        try:
            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Set dimension on first add
            if self.dimension is None:
                self.dimension = vectors_array.shape[1]
                self._create_index()
            
            # Add vectors
            start_id = 0 if self.index is None else self.index.ntotal
            self.index.add(vectors_array)
            
            # Store metadata
            for i, metadata in enumerate(metadata_list):
                vec_id = start_id + i
                self.metadata[str(vec_id)] = metadata
            
            self._save_metadata()
            logger.info(f"✅ Added {len(vectors)} vectors to FAISS index")
            
            return list(range(start_id, start_id + len(vectors)))
            
        except Exception as e:
            logger.error(f"❌ Error adding vectors: {e}")
            raise
    
    def _create_index(self):
        """Create FAISS index based on configured type"""
        try:
            if self.index_type.lower() == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            
            elif self.index_type.lower() == "ivf":
                # Use IVF with inner product
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                self.is_trained = False
            
            elif self.index_type.lower() == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            
            else:
                logger.warning(f"⚠️  Unknown index type {self.index_type}, using Flat")
                self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"✅ Created {self.index_type} index with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            raise
    
    def train_index(self, train_vectors: np.ndarray):
        """Train the index (required for IVF indices)
        
        Args:
            train_vectors: Training vectors as numpy array
        """
        if not isinstance(self.index, faiss.IndexIVFFlat):
            logger.info("ℹ️  Index type doesn't require training")
            return
        
        try:
            train_vectors = np.array(train_vectors, dtype=np.float32)
            self.index.train(train_vectors)
            self.is_trained = True
            logger.info("✅ Index trained successfully")
        except Exception as e:
            logger.error(f"❌ Error training index: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Search for similar vectors
        
        Args:
            query_vector: Query vector as numpy array
            k: Number of results to return
            
        Returns:
            Tuple of (vector_ids, distances)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("⚠️  Index is empty")
            return [], []
        
        try:
            query_vector = np.array([query_vector], dtype=np.float32)
            distances, indices = self.index.search(query_vector, k)
            
            # Unpack results
            vec_ids = indices[0].tolist()
            scores = distances[0].tolist()
            
            return vec_ids, scores
            
        except Exception as e:
            logger.error(f"❌ Error searching index: {e}")
            raise
    
    def search_with_metadata(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search and return results with metadata
        
        Args:
            query_vector: Query vector as numpy array
            k: Number of results to return
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        vec_ids, scores = self.search(query_vector, k)
        
        results = []
        for vec_id, score in zip(vec_ids, scores):
            result = {
                "id": int(vec_id),
                "score": float(score),
                "metadata": self.metadata.get(str(vec_id), {})
            }
            results.append(result)
        
        return results
    
    def save_index(self):
        """Save index to disk"""
        if self.index is None:
            logger.warning("⚠️  No index to save")
            return
        
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()
            logger.info(f"✅ Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"❌ Error saving index: {e}")
            raise
    
    def delete_vector(self, vector_id: int):
        """Delete a vector from the index (creates a new index without the vector)
        
        Args:
            vector_id: ID of vector to delete
        """
        if self.index is None:
            logger.warning("⚠️  Index is empty")
            return
        
        try:
            # FAISS doesn't support direct deletion, so we need to rebuild
            if str(vector_id) in self.metadata:
                del self.metadata[str(vector_id)]
            logger.info(f"✅ Vector {vector_id} marked for deletion")
        except Exception as e:
            logger.error(f"❌ Error deleting vector: {e}")
            raise
    
    def get_vector_count(self) -> int:
        """Get total number of vectors in index"""
        return self.index.ntotal if self.index else 0
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_vectors": self.get_vector_count(),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "index_path": self.index_path,
            "metadata_count": len(self.metadata)
        }
