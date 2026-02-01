"""
Embeddings Factory - Creates appropriate embeddings based on configuration
"""

from typing import Union, Dict, Any
from ..config.mode_selector import mode_selector, RAGMode, ComponentType
from ..config.hybrid_config import hybrid_config
import logging

logger = logging.getLogger(__name__)


def create_embeddings() -> Union['OpenAIEmbeddings', 'LocalEmbeddings']:
    """Create and return appropriate embeddings based on current mode
    
    Returns:
        Configured embeddings instance
        
    Raises:
        ValueError: If unsupported embeddings type is configured
        ImportError: If required dependencies are not installed
    """
    current_mode = mode_selector.mode
    embeddings_choice = mode_selector.get_component_choice(ComponentType.EMBEDDINGS)
    
    logger.info(f"🏭 Creating embeddings: {embeddings_choice} for mode: {current_mode.value}")
    
    if embeddings_choice == "openai":
        return _create_openai_embeddings()
    
    elif embeddings_choice == "local":
        return _create_local_embeddings()
    
    else:
        raise ValueError(f"Unsupported embeddings type: {embeddings_choice}")


def _create_openai_embeddings() -> 'OpenAIEmbeddings':
    """Create OpenAI embeddings instance
    
    Returns:
        Configured OpenAIEmbeddings instance
        
    Raises:
        ImportError: If openai package is not installed
        ValueError: If configuration is invalid
    """
    try:
        from .openai_embeddings import OpenAIEmbeddings
    except ImportError as e:
        logger.error("❌ OpenAI dependencies not installed. Run: pip install openai")
        raise ImportError("OpenAI embeddings requires 'openai' package") from e
    
    # Get configuration from hybrid config
    config = hybrid_config.get_embeddings_config()
    
    # Validate required configuration
    required_keys = ["api_key", "model"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing OpenAI embeddings configuration: {missing_keys}")
    
    logger.info("🌐 Creating OpenAI embeddings...")
    return OpenAIEmbeddings(config)


def _create_local_embeddings() -> 'LocalEmbeddings':
    """Create local embeddings instance
    
    Returns:
        Configured LocalEmbeddings instance
        
    Raises:
        ImportError: If required packages are not installed
        ValueError: If configuration is invalid
    """
    try:
        from .local_embeddings import LocalEmbeddings
    except ImportError as e:
        logger.error("❌ Local embeddings dependencies not installed. Run: pip install sentence-transformers")
        raise ImportError("Local embeddings requires 'sentence-transformers' package") from e
    
    # Get configuration from hybrid config
    config = hybrid_config.get_embeddings_config()
    
    # Validate required configuration
    required_keys = ["model_name", "device", "cache_dir"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing local embeddings configuration: {missing_keys}")
    
    logger.info("🤖 Creating local embeddings...")
    return LocalEmbeddings(config)


def get_available_embeddings() -> Dict[str, Dict[str, Any]]:
    """Get information about available embeddings options
    
    Returns:
        Dictionary with embeddings options and their availability
    """
    embeddings = {
        "openai": {
            "name": "OpenAI text-embedding-ada-002",
            "type": "online",
            "provider": "OpenAI",
            "description": "High-quality embeddings via OpenAI API",
            "available": False,
            "dimensions": 1536,
            "requirements": ["openai", "internet connection", "OpenAI API key"]
        },
        "local": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "local",
            "provider": "Sentence Transformers",
            "description": "Local lightweight embedding model",
            "available": False,
            "dimensions": 384,
            "requirements": ["sentence-transformers", "torch", "~100MB storage"]
        }
    }
    
    # Check OpenAI availability
    try:
        import openai
        from ..config.hybrid_config import hybrid_config
        config = hybrid_config.get_embeddings_config() if mode_selector.get_component_choice(ComponentType.EMBEDDINGS) == "openai" else None
        
        if config and config.get("api_key"):
            embeddings["openai"]["available"] = True
    except:
        pass
    
    # Check local embeddings availability
    try:
        import sentence_transformers
        import torch
        embeddings["local"]["available"] = True
    except ImportError:
        pass
    
    return embeddings


def validate_embeddings_config(embeddings_type: str) -> bool:
    """Validate embeddings configuration
    
    Args:
        embeddings_type: Type of embeddings to validate ("openai" or "local")
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        if embeddings_type == "openai":
            config = hybrid_config.get_embeddings_config()
            required_keys = ["api_key", "model"]
            return all(config.get(key) for key in required_keys)
        
        elif embeddings_type == "local":
            config = hybrid_config.get_embeddings_config()
            required_keys = ["model_name", "device", "cache_dir"]
            return all(config.get(key) for key in required_keys)
        
        else:
            return False
            
    except Exception:
        return False


def get_embeddings_dimension(embeddings_type: str = None) -> int:
    """Get embedding dimension for given type
    
    Args:
        embeddings_type: Type of embeddings (uses current mode if None)
        
    Returns:
        Embedding dimension
    """
    if embeddings_type is None:
        embeddings_type = mode_selector.get_component_choice(ComponentType.EMBEDDINGS)
    
    dimension_map = {
        "openai": 1536,  # text-embedding-ada-002
        "local": 384     # all-MiniLM-L6-v2
    }
    
    return dimension_map.get(embeddings_type, 384)


# Singleton instance for caching
_embeddings_instance = None
_current_embeddings_type = None


def get_embeddings(force_reload: bool = False) -> Union['OpenAIEmbeddings', 'LocalEmbeddings']:
    """Get or create embeddings instance with caching
    
    Args:
        force_reload: Force creation of new instance
        
    Returns:
        Cached or new embeddings instance
    """
    global _embeddings_instance, _current_embeddings_type
    
    current_embeddings_type = mode_selector.get_component_choice(ComponentType.EMBEDDINGS)
    
    # Return cached instance if same type and not forcing reload
    if (not force_reload and 
        _embeddings_instance is not None and 
        _current_embeddings_type == current_embeddings_type):
        return _embeddings_instance
    
    # Create new instance
    _embeddings_instance = create_embeddings()
    _current_embeddings_type = current_embeddings_type
    
    return _embeddings_instance


def clear_embeddings_cache():
    """Clear cached embeddings instance"""
    global _embeddings_instance, _current_embeddings_type
    
    if _embeddings_instance is not None:
        # Unload model if it's a local embeddings
        if hasattr(_embeddings_instance, 'unload_model'):
            _embeddings_instance.unload_model()
    
    _embeddings_instance = None
    _current_embeddings_type = None
    
    logger.info("🧹 Embeddings cache cleared")


def compare_embeddings_performance() -> Dict[str, Dict[str, Any]]:
    """Compare performance characteristics of available embeddings
    
    Returns:
        Dictionary with performance comparison
    """
    return {
        "openai": {
            "speed": "Fast (API call)",
            "quality": "High",
            "cost": "Pay per use",
            "offline": False,
            "memory_usage": "None (API)",
            "setup_time": "Instant",
            "dimensions": 1536,
            "best_for": "Production, high-quality results"
        },
        "local": {
            "speed": "Medium-Fast (GPU) / Slow (CPU)",
            "quality": "Good",
            "cost": "Free after setup",
            "offline": True,
            "memory_usage": "~100MB",
            "setup_time": "1-3 minutes (download)",
            "dimensions": 384,
            "best_for": "Privacy, offline use, cost optimization"
        }
    }