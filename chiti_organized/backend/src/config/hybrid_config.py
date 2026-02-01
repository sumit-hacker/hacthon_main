"""
Hybrid Configuration - Unified configuration interface for all modes
"""

from typing import Dict, Any
import os
import logging
from .mode_selector import mode_selector, ComponentType
from .local_config import get_local_config
from .online_config import get_online_config

logger = logging.getLogger(__name__)


class HybridConfig:
    """Manages configuration across online, local, and hybrid modes"""
    
    def __init__(self):
        """Initialize hybrid configuration"""
        self.mode = mode_selector.mode
        self.component_choices = mode_selector.config
        
        # Load appropriate configs
        self.local_config = None
        self.online_config = None
        
        logger.info(f"🔧 Initializing hybrid configuration for {self.mode.value} mode")

    def reload(self) -> None:
        """Reload mode and clear cached local/online configs.

        This should be called after updating environment variables at runtime.
        """
        self.mode = mode_selector.mode
        self.component_choices = mode_selector.config
        self.local_config = None
        self.online_config = None
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on current mode
        
        Returns:
            Dictionary with LLM configuration
        """
        llm_choice = mode_selector.get_component_choice(ComponentType.LLM)
        
        if llm_choice == "azure":
            if self.online_config is None:
                self.online_config = get_online_config()
            return {
                "endpoint": self.online_config.azure_openai_endpoint,
                "api_key": self.online_config.azure_openai_api_key,
                "api_version": self.online_config.azure_openai_api_version,
                "deployment_name": self.online_config.azure_openai_deployment_name,
                "model": self.online_config.azure_openai_model,
                "max_tokens": self.online_config.max_tokens,
                "temperature": self.online_config.temperature
            }
        
        elif llm_choice == "phi3":
            if self.local_config is None:
                self.local_config = get_local_config()
            return {
                "model_name": self.local_config.phi3_model_name,
                "device": self.local_config.phi3_device,
                "max_tokens": self.local_config.phi3_max_tokens,
                "temperature": self.local_config.phi3_temperature,
                "cache_dir": self.local_config.phi3_cache_dir
            }
        
        elif llm_choice == "tinyllama":
            if self.local_config is None:
                self.local_config = get_local_config()
            return {
                "model_name": self.local_config.tinyllama_model_name,
                "device": self.local_config.tinyllama_device,
                "max_tokens": self.local_config.tinyllama_max_tokens,
                "temperature": self.local_config.tinyllama_temperature,
                "cache_dir": self.local_config.tinyllama_cache_dir
            }
        
        else:
            raise ValueError(f"Unknown LLM choice: {llm_choice}")
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration based on current mode
        
        Returns:
            Dictionary with embeddings configuration
        """
        embeddings_choice = mode_selector.get_component_choice(ComponentType.EMBEDDINGS)
        
        if embeddings_choice == "openai":
            if self.online_config is None:
                self.online_config = get_online_config()
            return {
                "api_key": self.online_config.openai_api_key,
                "model": self.online_config.openai_embedding_model,
                "dimensions": self.online_config.openai_embedding_dimensions
            }
        
        elif embeddings_choice == "local":
            if self.local_config is None:
                self.local_config = get_local_config()
            return {
                "model_name": self.local_config.local_embedding_model,
                "device": self.local_config.local_embedding_device,
                "cache_dir": self.local_config.local_embedding_cache_dir,
                "dimensions": self.local_config.local_embedding_dimensions
            }
        
        else:
            raise ValueError(f"Unknown embeddings choice: {embeddings_choice}")
    
    def get_vectorstore_config(self) -> Dict[str, Any]:
        """Get vector store configuration based on current mode
        
        Returns:
            Dictionary with vector store configuration
        """
        vectorstore_choice = mode_selector.get_component_choice(ComponentType.VECTORSTORE)
        
        if vectorstore_choice == "qdrant":
            if self.online_config is None:
                self.online_config = get_online_config()
            return {
                "url": self.online_config.qdrant_url,
                "api_key": self.online_config.qdrant_api_key,
                "collection_name": self.online_config.qdrant_collection_name,
                "timeout": self.online_config.qdrant_timeout
            }
        
        elif vectorstore_choice == "faiss":
            if self.local_config is None:
                self.local_config = get_local_config()
            return {
                "index_path": self.local_config.faiss_index_path,
                "index_type": self.local_config.faiss_index_type,
                "nlist": self.local_config.faiss_nlist
            }
        
        else:
            raise ValueError(f"Unknown vector store choice: {vectorstore_choice}")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration based on current mode
        
        Returns:
            Dictionary with database configuration
        """
        database_choice = mode_selector.get_component_choice(ComponentType.DATABASE)
        
        if database_choice == "atlas":
            if self.online_config is None:
                self.online_config = get_online_config()
            return {
                "uri": self.online_config.mongodb_atlas_uri,
                "db_name": self.online_config.mongodb_atlas_db_name,
                "collection": self.online_config.mongodb_atlas_collection
            }
        
        elif database_choice == "local":
            if self.local_config is None:
                self.local_config = get_local_config()
            return {
                "uri": self.local_config.mongodb_local_uri,
                "db_name": self.local_config.mongodb_local_db_name,
                "collection": self.local_config.mongodb_local_collection
            }
        
        else:
            raise ValueError(f"Unknown database choice: {database_choice}")
    
    def get_general_config(self) -> Dict[str, Any]:
        """Get general configuration parameters
        
        Returns:
            Dictionary with general settings
        """
        if self.local_config is None:
            self.local_config = get_local_config()
        
        return {
            "max_tokens": self.local_config.max_tokens,
            "temperature": self.local_config.temperature,
            "top_k": self.local_config.top_k,
            "chunk_size": self.local_config.chunk_size,
            "chunk_overlap": self.local_config.chunk_overlap,
            "batch_size": self.local_config.batch_size,
            "max_workers": self.local_config.max_workers,
            "data_dir": self.local_config.data_dir,
            "logs_dir": self.local_config.logs_dir
        }
    
    def validate_all_configs(self) -> bool:
        """Validate all required configurations
        
        Returns:
            True if all configs are valid
        """
        try:
            # Validate mode selector
            if not mode_selector.validate_config():
                logger.error("❌ Invalid mode selector configuration")
                return False
            
            # Test loading all required configs
            self.get_llm_config()
            self.get_embeddings_config()
            self.get_vectorstore_config()
            self.get_database_config()
            self.get_general_config()
            
            logger.info("✅ All configurations validated successfully")
            return True
        
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False


# Global hybrid config instance
hybrid_config = HybridConfig()
