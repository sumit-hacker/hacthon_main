"""
OpenAI Embeddings Implementation for Online Mode
"""

from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class OpenAIEmbeddings:
    """OpenAI embeddings for online mode"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI embeddings client
        
        Args:
            config: Dictionary containing OpenAI embeddings configuration
        """
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.dimensions = config["dimensions"]
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"🌐 Initialized OpenAI Embeddings: {self.model}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error embedding text: {e}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            all_embeddings = []
            
            # Process in batches to avoid API limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"📊 Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            return np.array(all_embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error embedding texts: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query (same as embed_text for OpenAI)
        
        Args:
            query: Query to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.embed_text(query)
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings for documents (same as embed_texts for OpenAI)
        
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
        
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarities
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "similarity": float(similarities[idx])
            })
        
        return results
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def estimate_cost(self, texts: List[str]) -> Dict[str, Any]:
        """Estimate API cost for embedding texts
        
        Args:
            texts: List of texts to estimate cost for
            
        Returns:
            Dictionary with cost estimation
        """
        total_tokens = sum(self.estimate_tokens(text) for text in texts)
        
        # OpenAI text-embedding-ada-002 pricing: $0.0001 per 1K tokens
        cost_per_1k_tokens = 0.0001
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            "total_texts": len(texts),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "model": self.model
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": "OpenAI",
            "model": self.model,
            "dimensions": self.dimensions,
            "type": "online",
            "max_input_tokens": 8191  # For text-embedding-ada-002
        }
    
    def is_available(self) -> bool:
        """Check if the embeddings model is available
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Test with a simple embedding
            test_response = self.client.embeddings.create(
                input=["test"],
                model=self.model
            )
            return len(test_response.data) > 0
        except Exception as e:
            logger.error(f"❌ OpenAI embeddings not available: {e}")
            return False