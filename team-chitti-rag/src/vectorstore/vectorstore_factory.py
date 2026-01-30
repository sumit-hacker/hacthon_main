"""
Vector Store Factory - Creates appropriate vector store based on configuration
Supports: FAISS (local), Qdrant (online)
"""

from typing import Union, Dict, Any
from ..config.mode_selector import mode_selector, RAGMode, ComponentType
from ..config.hybrid_config import hybrid_config
import logging
import os

logger = logging.getLogger(__name__)


def create_vectorstore() -> Union['FAISSVectorStore', 'QdrantVectorStore']:
    """Create and return appropriate vector store based on current mode
    
    Returns:
        Configured vector store instance
        
    Raises:
        ValueError: If unsupported vector store type is configured
        ImportError: If required dependencies are not installed
    """
    current_mode = mode_selector.mode
    vectorstore_choice = mode_selector.get_component_choice(ComponentType.VECTORSTORE)
    
    logger.info(f"🏭 Creating vector store: {vectorstore_choice} for mode: {current_mode.value}")
    
    if vectorstore_choice == "faiss":
        return _create_faiss_vectorstore()
    
    elif vectorstore_choice == "qdrant":
        return _create_qdrant_vectorstore()
    
    else:
        raise ValueError(f"Unsupported vector store type: {vectorstore_choice}")


def _create_faiss_vectorstore() -> 'FAISSVectorStore':
    """Create FAISS vector store instance"""
    try:
        import faiss
    except ImportError as e:
        logger.error("❌ FAISS not installed. Run: pip install faiss-cpu")
        raise ImportError("FAISS vector store requires 'faiss-cpu' or 'faiss-gpu' package") from e
    
    config = hybrid_config.get_vectorstore_config()
    
    required_keys = ["index_path", "index_type"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing FAISS configuration: {missing_keys}")
    
    # Create index directory if it doesn't exist
    index_dir = os.path.dirname(config["index_path"])
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    
    logger.info("📦 Creating FAISS vector store...")
    
    # Import and create FAISS vector store
    try:
        from .faiss_vectorstore import FAISSVectorStore
        return FAISSVectorStore(config)
    except ImportError:
        logger.error("❌ FAISS vector store implementation not found")
        raise


def _create_qdrant_vectorstore() -> 'QdrantVectorStore':
    """Create Qdrant vector store instance"""
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        logger.error("❌ Qdrant client not installed. Run: pip install qdrant-client")
        raise ImportError("Qdrant vector store requires 'qdrant-client' package") from e
    
    config = hybrid_config.get_vectorstore_config()
    
    required_keys = ["url", "collection_name"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing Qdrant configuration: {missing_keys}")
    
    logger.info("☁️  Creating Qdrant vector store...")
    
    # Import and create Qdrant vector store
    try:
        from .qdrant_vectorstore import QdrantVectorStore
        return QdrantVectorStore(config)
    except ImportError:
        logger.error("❌ Qdrant vector store implementation not found")
        raise


# Singleton instance for caching
_vectorstore_instance = None
_current_vectorstore_type = None


def get_vectorstore() -> Union['FAISSVectorStore', 'QdrantVectorStore']:
    """Get or create vector store instance (singleton pattern)
    
    Returns:
        Cached or newly created vector store instance
    """
    global _vectorstore_instance, _current_vectorstore_type
    
    current_vectorstore_type = mode_selector.get_component_choice(ComponentType.VECTORSTORE)
    
    # Create new instance if type changed or not cached
    if _vectorstore_instance is None or _current_vectorstore_type != current_vectorstore_type:
        _vectorstore_instance = create_vectorstore()
        _current_vectorstore_type = current_vectorstore_type
    
    return _vectorstore_instance


def get_available_vectorstores() -> Dict[str, Dict[str, Any]]:
    """Get information about available vector store implementations
    
    Returns:
        Dictionary with vector store availability information
    """
    vectorstores = {
        "faiss": {
            "name": "FAISS (Local)",
            "type": "local",
            "available": _check_package("faiss"),
            "requirements": "faiss-cpu or faiss-gpu",
            "description": "Fast local vector similarity search"
        },
        "qdrant": {
            "name": "Qdrant (Cloud)",
            "type": "online",
            "available": _check_package("qdrant_client"),
            "requirements": "qdrant-client",
            "description": "Scalable cloud-based vector database"
        }
    }
    
    return vectorstores


def _check_package(package_name: str) -> bool:
    """Check if a package is installed
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if installed, False otherwise
    """
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def reset_vectorstore_cache():
    """Reset the cached vector store instance (useful for testing)"""
    global _vectorstore_instance, _current_vectorstore_type
    _vectorstore_instance = None
    _current_vectorstore_type = None
