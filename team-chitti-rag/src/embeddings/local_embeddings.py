"""
Local Sentence Transformers Embeddings for Local/Offline Mode
"""

from typing import List, Dict, Any
import numpy as np
import torch
import os
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalEmbeddings:
    """Local sentence transformers embeddings for offline mode"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize local embeddings model
        
        Args:
            config: Dictionary containing local embeddings configuration
        """
        self.model_name = config["model_name"]
        self.device = config["device"]
        self.cache_dir = config["cache_dir"]
        self.dimensions = config["dimensions"]
        
        self.model = None
        self.model_loaded = False
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"🤖 Initializing local embeddings: {self.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info("📥 Loading sentence transformer model... (this may take a few minutes)")

            # Prefer offline/local loading to avoid timeouts/rate limits in Codespaces.
            # If the model isn't present in cache, we fall back to online loading.
            prefer_local_only = os.getenv("EMBEDDINGS_LOCAL_ONLY", "true").lower() in {"1", "true", "yes"}

            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self.device,
                    local_files_only=prefer_local_only,
                )
            except Exception:
                if not prefer_local_only:
                    raise
                logger.warning("⚠️  Local-only embeddings load failed; retrying with network allowed...")
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=self.device,
                    local_files_only=False,
                )
            
            self.model_loaded = True
            
            # Get actual dimensions from model
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.dimensions = test_embedding.shape[1]
            
            logger.info(f"✅ Local embeddings model loaded successfully on {self.device}")
            logger.info(f"📏 Embedding dimensions: {self.dimensions}")
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings model: {e}")
            self.model_loaded = False
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot generate embeddings.")
        
        try:
            embedding = self.model.encode([text], show_progress_bar=False)
            return embedding[0].astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error embedding text: {e}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot generate embeddings.")
        
        try:
            # Use sentence transformer's batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error embedding texts: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query
        
        Args:
            query: Query to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.embed_text(query)
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.embed_texts(documents)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings
        
        Returns:
            Embedding dimension
        """
        return self.dimensions
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         document_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar documents to query
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Array of document embeddings
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with indices and similarity scores
        """
        if document_embeddings.size == 0:
            return []
        
        # Use sentence transformer's utility function for similarity
        from sentence_transformers.util import cos_sim
        
        # Calculate cosine similarities
        similarities = cos_sim(query_embedding, document_embeddings)[0]
        
        # Get top k indices
        top_results = torch.topk(similarities, k=min(top_k, len(similarities)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "index": int(idx),
                "similarity": float(score)
            })
        
        return results
    
    def encode_corpus(self, corpus: List[str], batch_size: int = 32, 
                     show_progress: bool = True) -> np.ndarray:
        """Encode a large corpus efficiently
        
        Args:
            corpus: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot generate embeddings.")
        
        try:
            embeddings = self.model.encode(
                corpus,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error encoding corpus: {e}")
            raise
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length supported by the model
        
        Returns:
            Maximum sequence length in tokens
        """
        if self.model_loaded and hasattr(self.model, 'max_seq_length'):
            return self.model.max_seq_length
        else:
            return 512  # Default for many sentence transformer models
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to fit model's max sequence length
        
        Args:
            text: Text to truncate
            max_length: Maximum length (uses model default if None)
            
        Returns:
            Truncated text
        """
        if max_length is None:
            max_length = self.get_max_sequence_length()
        
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_length * 4
        
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars-3] + "..."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information
        
        Returns:
            Dictionary with model information
        """
        info = {
            "provider": "Sentence Transformers",
            "model": self.model_name,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "dimensions": self.dimensions,
            "type": "local",
            "loaded": self.model_loaded
        }
        
        if self.model_loaded:
            info.update({
                "max_seq_length": self.get_max_sequence_length(),
                "pooling_mode": str(self.model._modules.get('1', 'unknown'))  # Pooling layer info
            })
        
        return info
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available
        
        Returns:
            True if model is available, False otherwise
        """
        return self.model_loaded and self.model is not None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information
        
        Returns:
            Dictionary with memory usage information
        """
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        return {
            "gpu_available": True,
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
            "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,  # MB
            "gpu_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None
        }
    
    def unload_model(self):
        """Unload the model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.model_loaded = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🗑️  Local embeddings model unloaded from memory")