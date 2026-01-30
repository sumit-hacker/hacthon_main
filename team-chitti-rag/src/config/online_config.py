"""
Online Configuration - Azure OpenAI + Qdrant + MongoDB Atlas
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class OnlineConfig(BaseSettings):
    """Configuration for online/cloud components"""
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="2024-02-01", env="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment_name: str = Field(default="gpt-4", env="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_model: str = Field(default="gpt-4", env="AZURE_OPENAI_MODEL")
    
    # OpenAI Embeddings Configuration  
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=1536, env="OPENAI_EMBEDDING_DIMENSIONS")
    
    # Qdrant Configuration
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="team_chitti_documents", env="QDRANT_COLLECTION")
    qdrant_timeout: int = Field(default=30, env="QDRANT_TIMEOUT")
    
    # MongoDB Atlas Configuration
    mongodb_atlas_uri: str = Field(..., env="MONGODB_ATLAS_URI")
    mongodb_atlas_db_name: str = Field(default="team_chitti_rag", env="MONGODB_ATLAS_DB_NAME")
    mongodb_atlas_collection: str = Field(default="documents", env="MONGODB_ATLAS_COLLECTION")
    
    # General Settings
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    top_k: int = Field(default=5, env="TOP_K")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


def get_online_config() -> OnlineConfig:
    """Get validated online configuration"""
    try:
        config = OnlineConfig()
        print("✅ Online configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Failed to load online configuration: {e}")
        print("💡 Make sure all required environment variables are set for online mode")
        raise


def validate_online_dependencies():
    """Validate that all required packages for online mode are available"""
    required_packages = [
        "openai",
        "qdrant-client", 
        "pymongo",
        "azure-identity"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages for online mode: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True