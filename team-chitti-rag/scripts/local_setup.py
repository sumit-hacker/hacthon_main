#!/usr/bin/env python3
"""
Local RAG Setup Script - Initialize local mode with TinyLlama + FAISS + Local Embeddings
Standalone script to download and test all components
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Ensure repo root is importable so `import src.*` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print setup banner"""
    print("\n" + "=" * 70)
    print("🚀 Team Chitti RAG - Local Setup (TinyLlama + FAISS + Local Embeddings)")
    print("=" * 70)
    print("This script will download and test all local components")
    print("=" * 70 + "\n")


def check_dependencies() -> bool:
    """Check if all required packages are installed
    
    Returns:
        True if all dependencies are available
    """
    print("📦 Checking dependencies...")
    print("-" * 70)
    
    required_packages = {
        "torch": "PyTorch - Deep learning framework",
        "transformers": "HuggingFace Transformers - LLM library",
        "sentence_transformers": "Sentence Transformers - Embedding models",
        "faiss": "FAISS - Vector similarity search",
        "pymongo": "MongoDB driver - Local database",
        "numpy": "NumPy - Numerical computing",
        "pydantic": "Pydantic - Data validation"
    }
    
    missing_packages = []
    
    for package_name, description in required_packages.items():
        try:
            __import__(package_name.replace("-", "_"))
            print(f"✅ {package_name:20} - {description}")
        except ImportError:
            print(f"❌ {package_name:20} - {description}")
            missing_packages.append(package_name)
    
    print("-" * 70)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("\n💡 Install with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!\n")
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    print("-" * 70)
    
    directories = [
        "./models/tinyllama",
        "./models/embeddings",
        "./data/vector_store",
        "./logs",
        "./data/documents"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    print("-" * 70 + "\n")


def test_tinyllama():
    """Test TinyLlama LLM"""
    print("🤖 Testing TinyLlama LLM...")
    print("-" * 70)
    
    try:
        from src.llm.tinyllama_local_llm import TinyLlamaLocalLLM
        from src.config.local_config import get_local_config
        
        config = get_local_config()
        llm_config = {
            "model_name": config.tinyllama_model_name,
            "device": "cpu",  # Force CPU for testing
            "max_tokens": 100,
            "temperature": 0.1,
            "cache_dir": config.tinyllama_cache_dir
        }
        
        print("📥 Loading TinyLlama model...")
        llm = TinyLlamaLocalLLM(llm_config)
        
        print("✅ TinyLlama loaded successfully!")
        
        # Test generation
        print("\n📝 Testing generation...")
        test_prompt = "What is machine learning? Answer in 1-2 sentences."
        response = llm.generate(test_prompt, max_tokens=50)
        
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")
        print("-" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ TinyLlama test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings():
    """Test local embeddings"""
    print("🔗 Testing Local Embeddings...")
    print("-" * 70)
    
    try:
        from src.embeddings.local_embeddings import LocalEmbeddings
        from src.config.local_config import get_local_config
        
        config = get_local_config()
        embeddings_config = {
            "model_name": config.local_embedding_model,
            "device": "cpu",  # Force CPU for testing
            "cache_dir": config.local_embedding_cache_dir,
            "dimensions": config.local_embedding_dimensions
        }
        
        print("📥 Loading embedding model...")
        embeddings = LocalEmbeddings(embeddings_config)
        
        print("✅ Embeddings model loaded successfully!")
        
        # Test embedding
        print("\n📝 Testing embedding generation...")
        test_texts = [
            "The capital of India is New Delhi.",
            "Education is the most powerful weapon.",
            "Machine learning is a subset of AI."
        ]
        
        for text in test_texts:
            embedding = embeddings.embed_text(text)
            print(f"✅ Text: '{text[:50]}...' -> Dimension: {embedding.shape}")
        
        print("-" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faiss():
    """Test FAISS vector store"""
    print("🗂️  Testing FAISS Vector Store...")
    print("-" * 70)
    
    try:
        import numpy as np
        from src.vectorstore.faiss_vectorstore import FAISSVectorStore
        from src.config.local_config import get_local_config
        
        config = get_local_config()
        vectorstore_config = {
            "index_path": config.faiss_index_path,
            "index_type": "Flat",
            "nlist": 100
        }
        
        print("🏭 Creating FAISS vector store...")
        vectorstore = FAISSVectorStore(vectorstore_config)
        
        # Add sample vectors
        print("📥 Adding sample vectors...")
        sample_vectors = [
            np.random.rand(384).astype(np.float32) for _ in range(10)
        ]
        metadata_list = [
            {"id": i, "text": f"Sample document {i}"} for i in range(10)
        ]
        
        vec_ids = vectorstore.add_vectors(sample_vectors, metadata_list)
        print(f"✅ Added {len(vec_ids)} vectors")
        
        # Test search
        print("\n🔍 Testing search...")
        query_vector = sample_vectors[0]
        results = vectorstore.search_with_metadata(query_vector, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.4f}, Metadata: {result['metadata']}")
        
        # Save index
        print("\n💾 Saving index...")
        vectorstore.save_index()
        
        stats = vectorstore.get_index_stats()
        print(f"✅ Index saved - Stats: {stats}")
        
        print("-" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mongodb():
    """Test local MongoDB connection"""
    print("💾 Testing Local MongoDB...")
    print("-" * 70)
    
    try:
        from pymongo import MongoClient
        
        uri = "mongodb://localhost:27017"
        print(f"Connecting to: {uri}")
        
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        
        # Try to ping
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Check databases
        dbs = client.list_database_names()
        print(f"Available databases: {dbs}")
        
        client.close()
        print("-" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"⚠️  MongoDB test failed: {e}")
        print("💡 Make sure MongoDB is running: mongod")
        print("-" * 70 + "\n")
        
        return False


def test_configuration():
    """Test configuration system"""
    print("⚙️  Testing Configuration System...")
    print("-" * 70)
    
    try:
        from src.config.mode_selector import mode_selector, ComponentType
        from src.config.hybrid_config import hybrid_config
        
        # Check mode
        print(f"Current Mode: {mode_selector.mode.value}")
        print(f"LLM: {mode_selector.get_component_choice(ComponentType.LLM)}")
        print(f"Embeddings: {mode_selector.get_component_choice(ComponentType.EMBEDDINGS)}")
        print(f"Vector Store: {mode_selector.get_component_choice(ComponentType.VECTORSTORE)}")
        print(f"Database: {mode_selector.get_component_choice(ComponentType.DATABASE)}")
        
        # Validate config
        if hybrid_config.validate_all_configs():
            print("\n✅ Configuration is valid!")
        else:
            print("\n❌ Configuration validation failed")
            return False
        
        print("-" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup local RAG components")
    parser.add_argument("--skip-tests", action="store_true", help="Skip component tests")
    parser.add_argument("--skip-llm", action="store_true", help="Skip TinyLlama download")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embeddings download")
    
    args = parser.parse_args()
    
    print_banner()
    
    # 1. Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies")
        sys.exit(1)
    
    # 2. Create directories
    create_directories()
    
    # 3. Test configuration
    if not test_configuration():
        print("❌ Configuration test failed")
        sys.exit(1)
    
    if args.skip_tests:
        print("⏭️  Skipping component tests")
        return
    
    # 4. Test components
    results = {
        "TinyLlama": True,
        "Embeddings": True,
        "FAISS": True,
        "MongoDB": True
    }
    
    if not args.skip_llm:
        results["TinyLlama"] = test_tinyllama()
    
    if not args.skip_embeddings:
        results["Embeddings"] = test_embeddings()
    
    results["FAISS"] = test_faiss()
    results["MongoDB"] = test_mongodb()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SETUP SUMMARY")
    print("=" * 70)
    
    for component, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {component}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✨ Setup complete! You're ready to use Team Chitti RAG")
        print("\n💡 Next steps:")
        print("   1. Process your documents: python -m src.database.document_processor")
        print("   2. Create embeddings and index them")
        print("   3. Run the application: python main.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
