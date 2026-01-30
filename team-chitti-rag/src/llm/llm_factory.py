"""
LLM Factory - Creates appropriate LLM based on configuration
Supports: Azure OpenAI, Phi-3, TinyLlama
"""

from typing import Union, Dict, Any
from ..config.mode_selector import mode_selector, RAGMode, ComponentType
from ..config.hybrid_config import hybrid_config
import logging

logger = logging.getLogger(__name__)


def create_llm() -> Union['AzureOpenAILLM', 'Phi3LocalLLM', 'TinyLlamaLocalLLM']:
    """Create and return appropriate LLM based on current mode
    
    Returns:
        Configured LLM instance
        
    Raises:
        ValueError: If unsupported LLM type is configured
        ImportError: If required dependencies are not installed
    """
    current_mode = mode_selector.mode
    llm_choice = mode_selector.get_component_choice(ComponentType.LLM)
    
    logger.info(f"🏭 Creating LLM: {llm_choice} for mode: {current_mode.value}")
    
    if llm_choice == "azure":
        return _create_azure_openai_llm()
    
    elif llm_choice == "phi3":
        return _create_phi3_local_llm()
    
    elif llm_choice == "tinyllama":
        return _create_tinyllama_local_llm()
    
    else:
        raise ValueError(f"Unsupported LLM type: {llm_choice}")


def _create_azure_openai_llm() -> 'AzureOpenAILLM':
    """Create Azure OpenAI LLM instance"""
    try:
        from .azure_openai_llm import AzureOpenAILLM
    except ImportError as e:
        logger.error("❌ Azure OpenAI dependencies not installed. Run: pip install openai")
        raise ImportError("Azure OpenAI LLM requires 'openai' package") from e
    
    config = hybrid_config.get_llm_config()
    
    required_keys = ["endpoint", "api_key", "deployment_name"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing Azure OpenAI configuration: {missing_keys}")
    
    logger.info("🌐 Creating Azure OpenAI LLM...")
    return AzureOpenAILLM(config)


def _create_phi3_local_llm() -> 'Phi3LocalLLM':
    """Create Phi-3 Mini local LLM instance"""
    try:
        from .phi3_local_llm import Phi3LocalLLM
    except ImportError as e:
        logger.error("❌ Phi-3 dependencies not installed. Run: pip install torch transformers")
        raise ImportError("Phi-3 LLM requires 'torch' and 'transformers' packages") from e
    
    config = hybrid_config.get_llm_config()
    
    required_keys = ["model_name", "device", "max_tokens", "temperature", "cache_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing Phi-3 configuration: {missing_keys}")
    
    logger.info("🤖 Creating Phi-3 Mini local LLM...")
    return Phi3LocalLLM(config)


def _create_tinyllama_local_llm() -> 'TinyLlamaLocalLLM':
    """Create TinyLlama local LLM instance"""
    try:
        from .tinyllama_local_llm import TinyLlamaLocalLLM
    except ImportError as e:
        logger.error("❌ TinyLlama dependencies not installed. Run: pip install torch transformers")
        raise ImportError("TinyLlama LLM requires 'torch' and 'transformers' packages") from e
    
    config = hybrid_config.get_llm_config()
    
    required_keys = ["model_name", "device", "max_tokens", "temperature", "cache_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing TinyLlama configuration: {missing_keys}")
    
    logger.info("🚀 Creating TinyLlama local LLM (lightweight & fast)...")
    return TinyLlamaLocalLLM(config)


# Singleton instance for caching
_llm_instance = None
_current_llm_type = None


def get_llm() -> Union['AzureOpenAILLM', 'Phi3LocalLLM', 'TinyLlamaLocalLLM']:
    """Get or create LLM instance (singleton pattern)
    
    Returns:
        Cached or newly created LLM instance
    """
    global _llm_instance, _current_llm_type
    
    current_llm_type = mode_selector.get_component_choice(ComponentType.LLM)
    
    # Create new instance if type changed or not cached
    if _llm_instance is None or _current_llm_type != current_llm_type:
        _llm_instance = create_llm()
        _current_llm_type = current_llm_type
    
    return _llm_instance


def get_available_llms() -> Dict[str, Dict[str, Any]]:
    """Get information about available LLM implementations
    
    Returns:
        Dictionary with LLM availability information
    """
    llms = {
        "azure": {
            "name": "Azure OpenAI (GPT-4)",
            "type": "online",
            "available": _check_package("openai"),
            "requirements": "openai",
            "description": "Enterprise-grade GPT-4 via Azure"
        },
        "phi3": {
            "name": "Phi-3 Mini (3.8B)",
            "type": "local",
            "available": _check_package("torch") and _check_package("transformers"),
            "requirements": "torch, transformers",
            "description": "Lightweight local LLM (requires ~8GB GPU or 16GB RAM)"
        },
        "tinyllama": {
            "name": "TinyLlama (1.1B)",
            "type": "local",
            "available": _check_package("torch") and _check_package("transformers"),
            "requirements": "torch, transformers",
            "description": "Ultra-lightweight local LLM (requires ~4GB RAM)"
        }
    }
    
    return llms


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


def reset_llm_cache():
    """Reset the cached LLM instance (useful for testing)"""
    global _llm_instance, _current_llm_type
    _llm_instance = None
    _current_llm_type = None
