"""
Local Configuration - Phi-3 Mini + Local Embeddings + FAISS + MongoDB Local
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


def _clean_inline_comment(value: str) -> str:
    """Best-effort cleanup for dotenv values that may include inline comments."""
    if not isinstance(value, str):
        return value
    # Split on first '#', then take first whitespace-separated token.
    value = value.split('#', 1)[0].strip()
    return value.split()[0] if value else value


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device selection into 'cuda' (if available) else 'cpu'."""
    device = _clean_inline_comment(device).lower() if device else "cpu"
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class LocalConfig(BaseSettings):
    """Configuration for local/offline components"""
    
    # Phi-3 Mini Configuration (3.8B parameters, ~5GB)
    phi3_model_name: str = Field(default="microsoft/Phi-3-mini-4k-instruct", env="PHI3_MODEL_NAME")
    phi3_device: str = Field(default="auto", env="PHI3_DEVICE")  # auto, cpu, cuda
    phi3_max_tokens: int = Field(default=2000, env="PHI3_MAX_TOKENS")
    phi3_temperature: float = Field(default=0.1, env="PHI3_TEMPERATURE")
    phi3_cache_dir: str = Field(default="./models/phi3", env="PHI3_CACHE_DIR")
    
    # TinyLlama Configuration (1.1B parameters, ~1GB - for low resource environments)
    tinyllama_model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", env="TINYLLAMA_MODEL_NAME")
    tinyllama_device: str = Field(default="auto", env="TINYLLAMA_DEVICE")  # auto, cpu, cuda
    tinyllama_max_tokens: int = Field(default=2000, env="TINYLLAMA_MAX_TOKENS")
    tinyllama_temperature: float = Field(default=0.1, env="TINYLLAMA_TEMPERATURE")
    tinyllama_cache_dir: str = Field(default="./models/tinyllama", env="TINYLLAMA_CACHE_DIR")
    
    # Local Embeddings Configuration (Sentence Transformers)
    local_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="LOCAL_EMBEDDING_MODEL")
    local_embedding_device: str = Field(default="cpu", env="LOCAL_EMBEDDING_DEVICE")
    local_embedding_cache_dir: str = Field(default="./models/embeddings", env="LOCAL_EMBEDDING_CACHE_DIR")
    local_embedding_dimensions: int = Field(default=384, env="LOCAL_EMBEDDING_DIMENSIONS")
    
    # FAISS Configuration
    faiss_index_path: str = Field(default="./data/faiss_index", env="FAISS_INDEX_PATH")
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")  # IndexFlatIP, IndexIVFFlat
    faiss_nlist: int = Field(default=100, env="FAISS_NLIST")  # For IVF indices
    
    # MongoDB Local Configuration
    mongodb_local_uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_LOCAL_URI")
    mongodb_local_db_name: str = Field(default="team_chitti_rag_local", env="MONGODB_LOCAL_DB_NAME")
    mongodb_local_collection: str = Field(default="documents", env="MONGODB_LOCAL_COLLECTION")
    
    # General Settings
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    top_k: int = Field(default=5, env="TOP_K")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Performance Settings
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    
    # Storage Settings
    data_dir: str = Field(default="./data", env="DATA_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


def get_local_config() -> LocalConfig:
    """Get validated local configuration"""
    try:
        config = LocalConfig()

        # Normalize device fields for local execution
        config.phi3_device = _resolve_device(config.phi3_device)
        config.tinyllama_device = _resolve_device(config.tinyllama_device)
        config.local_embedding_device = _resolve_device(config.local_embedding_device)

        # Ensure FAISS index path has a stable extension for sidecar metadata.
        if config.faiss_index_path and not str(config.faiss_index_path).endswith(".faiss"):
            config.faiss_index_path = str(config.faiss_index_path) + ".faiss"
        
        # Create necessary directories
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.logs_dir, exist_ok=True)
        os.makedirs(config.phi3_cache_dir, exist_ok=True)
        os.makedirs(config.tinyllama_cache_dir, exist_ok=True)
        os.makedirs(config.local_embedding_cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config.faiss_index_path), exist_ok=True)
        
        print("✅ Local configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Failed to load local configuration: {e}")
        raise


def validate_local_dependencies():
    """Validate that all required packages for local mode are available"""
    required_packages = [
        "transformers",
        "torch", 
        "sentence_transformers",
        "faiss_cpu",  # or faiss-gpu
        "pymongo"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "faiss_cpu":
                __import__("faiss")
            else:
                __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages for local mode: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def check_system_resources():
    """Check if system has enough resources for local mode"""
    import psutil
    
    # Check available RAM (recommend 8GB+ for Phi-3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    if available_ram_gb < 4:
        print(f"⚠️  Warning: Low RAM ({available_ram_gb:.1f}GB). Recommend 8GB+ for optimal performance")
    
    # Check disk space
    disk_usage = psutil.disk_usage('./')
    free_space_gb = disk_usage.free / (1024**3)
    if free_space_gb < 10:
        print(f"⚠️  Warning: Low disk space ({free_space_gb:.1f}GB). Need ~10GB for models and data")
    
    return True