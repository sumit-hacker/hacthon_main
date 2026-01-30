"""
Database Factory - Creates appropriate database based on configuration
"""

from typing import Union, Dict, Any, List
from ..config.mode_selector import mode_selector, RAGMode, ComponentType
from ..config.hybrid_config import hybrid_config
import logging

logger = logging.getLogger(__name__)


def create_database() -> Union['MongoAtlasDB', 'MongoLocalDB']:
    """Create and return appropriate database based on current mode
    
    Returns:
        Configured database instance
        
    Raises:
        ValueError: If unsupported database type is configured
        ImportError: If required dependencies are not installed
    """
    current_mode = mode_selector.mode
    database_choice = mode_selector.get_component_choice(ComponentType.DATABASE)
    
    logger.info(f"🏭 Creating database: {database_choice} for mode: {current_mode.value}")
    
    if database_choice == "atlas":
        return _create_mongo_atlas_db()
    
    elif database_choice == "local":
        return _create_mongo_local_db()
    
    else:
        raise ValueError(f"Unsupported database type: {database_choice}")


def _create_mongo_atlas_db() -> 'MongoAtlasDB':
    """Create MongoDB Atlas database instance"""
    try:
        from .mongo_atlas_db import MongoAtlasDB
    except ImportError as e:
        logger.error("❌ MongoDB Atlas dependencies not installed. Run: pip install pymongo motor")
        raise ImportError("MongoDB Atlas requires 'pymongo' and 'motor' packages") from e
    
    config = hybrid_config.get_database_config()
    
    required_keys = ["uri", "db_name", "collection"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing MongoDB Atlas configuration: {missing_keys}")
    
    logger.info("🌐 Creating MongoDB Atlas database...")
    return MongoAtlasDB(config)


def _create_mongo_local_db() -> 'MongoLocalDB':
    """Create local MongoDB database instance"""
    try:
        from .mongo_local_db import MongoLocalDB
    except ImportError as e:
        logger.error("❌ MongoDB dependencies not installed. Run: pip install pymongo")
        raise ImportError("Local MongoDB requires 'pymongo' package") from e
    
    config = hybrid_config.get_database_config()
    
    required_keys = ["uri", "db_name", "collection"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing local MongoDB configuration: {missing_keys}")
    
    logger.info("🤖 Creating local MongoDB database...")
    return MongoLocalDB(config)


# Singleton instance for caching
_database_instance = None
_current_database_type = None


def get_database(force_reload: bool = False) -> Union['MongoAtlasDB', 'MongoLocalDB']:
    """Get or create database instance with caching"""
    global _database_instance, _current_database_type
    
    current_database_type = mode_selector.get_component_choice(ComponentType.DATABASE)
    
    if (not force_reload and 
        _database_instance is not None and 
        _current_database_type == current_database_type):
        return _database_instance
    
    _database_instance = create_database()
    _current_database_type = current_database_type
    
    return _database_instance


def clear_database_cache():
    """Clear cached database instance"""
    global _database_instance, _current_database_type
    
    _database_instance = None
    _current_database_type = None
    
    logger.info("🧹 Database cache cleared")