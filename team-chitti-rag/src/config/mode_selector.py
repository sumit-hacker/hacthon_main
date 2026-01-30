"""
Mode Selector - Determines which components to use based on configuration
Supports: online, local, hybrid modes
"""

from enum import Enum
from typing import Dict, Any
import os
from dataclasses import dataclass


class RAGMode(Enum):
    """Available RAG operation modes"""
    ONLINE = "online"      # Azure OpenAI + Qdrant + MongoDB Atlas
    LOCAL = "local"        # Phi-3 + FAISS + MongoDB Local  
    HYBRID = "hybrid"      # Mix and match components


class ComponentType(Enum):
    """Types of components that can be configured"""
    LLM = "llm"
    EMBEDDINGS = "embeddings" 
    VECTORSTORE = "vectorstore"
    DATABASE = "database"


@dataclass
class ComponentChoice:
    """Represents a choice for each component"""
    llm: str           # "azure" or "phi3"
    embeddings: str    # "openai" or "local"
    vectorstore: str   # "qdrant" or "faiss"
    database: str      # "atlas" or "local"


class ModeSelector:
    """Selects appropriate components based on mode configuration"""
    
    # Predefined component combinations for each mode
    MODE_CONFIGS = {
        RAGMode.ONLINE: ComponentChoice(
            llm="azure",
            embeddings="openai", 
            vectorstore="qdrant",
            database="atlas"
        ),
        RAGMode.LOCAL: ComponentChoice(
            llm="tinyllama",  # Default to TinyLlama for lower resource usage
            embeddings="local",
            vectorstore="faiss", 
            database="local"
        )
    }
    
    def __init__(self, mode: RAGMode = None, custom_config: ComponentChoice = None):
        """
        Initialize mode selector
        
        Args:
            mode: Predefined mode (online/local)
            custom_config: Custom component configuration for hybrid mode
        """
        self.mode = mode or self._get_mode_from_env()
        
        if self.mode == RAGMode.HYBRID:
            if not custom_config:
                custom_config = self._get_hybrid_config_from_env()
            self.config = custom_config
        else:
            self.config = self.MODE_CONFIGS[self.mode]

    def reload_from_env(self) -> None:
        """Reload mode + component selection from environment variables.

        Important: this mutates the existing instance (instead of replacing it)
        so any modules that imported `mode_selector` keep working.
        """
        self.mode = self._get_mode_from_env()

        if self.mode == RAGMode.HYBRID:
            self.config = self._get_hybrid_config_from_env()
        else:
            self.config = self.MODE_CONFIGS[self.mode]
    
    def _get_mode_from_env(self) -> RAGMode:
        """Get mode from environment variable"""
        mode_str = os.getenv("RAG_MODE", "local").lower()
        try:
            return RAGMode(mode_str)
        except ValueError:
            print(f"Invalid RAG_MODE: {mode_str}. Defaulting to LOCAL mode.")
            return RAGMode.LOCAL
    
    def _get_hybrid_config_from_env(self) -> ComponentChoice:
        """Get hybrid configuration from environment variables"""
        return ComponentChoice(
            llm=os.getenv("RAG_LLM", "tinyllama").lower(),  # Default to tinyllama for lower resource usage
            embeddings=os.getenv("RAG_EMBEDDINGS", "local").lower(),
            vectorstore=os.getenv("RAG_VECTORSTORE", "faiss").lower(), 
            database=os.getenv("RAG_DATABASE", "local").lower()
        )
    
    def get_component_choice(self, component_type: ComponentType) -> str:
        """Get the chosen implementation for a component type"""
        component_map = {
            ComponentType.LLM: self.config.llm,
            ComponentType.EMBEDDINGS: self.config.embeddings,
            ComponentType.VECTORSTORE: self.config.vectorstore,
            ComponentType.DATABASE: self.config.database
        }
        return component_map[component_type]
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current mode and configuration"""
        return {
            "mode": self.mode.value,
            "llm": self.config.llm,
            "embeddings": self.config.embeddings,
            "vectorstore": self.config.vectorstore, 
            "database": self.config.database,
            "is_fully_online": self._is_fully_online(),
            "is_fully_local": self._is_fully_local(),
            "is_hybrid": self.mode == RAGMode.HYBRID
        }
    
    def _is_fully_online(self) -> bool:
        """Check if all components are online/cloud-based"""
        return (self.config.llm == "azure" and 
                self.config.embeddings == "openai" and
                self.config.vectorstore == "qdrant" and
                self.config.database == "atlas")
    
    def _is_fully_local(self) -> bool:
        """Check if all components are local/offline"""
        return (self.config.llm in ["phi3", "tinyllama"] and
                self.config.embeddings == "local" and 
                self.config.vectorstore == "faiss" and
                self.config.database == "local")
    
    def validate_config(self) -> bool:
        """Validate that the current configuration is supported"""
        valid_choices = {
            "llm": ["azure", "phi3", "tinyllama"],  # Support both local LLM options
            "embeddings": ["openai", "local"], 
            "vectorstore": ["qdrant", "faiss"],
            "database": ["atlas", "local"]
        }
        
        config_dict = {
            "llm": self.config.llm,
            "embeddings": self.config.embeddings,
            "vectorstore": self.config.vectorstore,
            "database": self.config.database
        }
        
        for component, choice in config_dict.items():
            if choice not in valid_choices[component]:
                print(f"Invalid {component} choice: {choice}")
                return False
        
        return True


# Global mode selector instance
mode_selector = ModeSelector()