"""
Team Chitti RAG - Main Application Entry Point
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path so `import src.*` works
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.mode_selector import mode_selector, RAGMode
from src.config.hybrid_config import hybrid_config
from src.llm.llm_factory import get_llm, get_available_llms
from src.embeddings.embeddings_factory import get_embeddings, get_available_embeddings
from src.database.document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🚀 Team Chitti RAG - Modular Q&A System")
    print("=" * 60)
    print("Modes: Online | Local | Hybrid")
    print("Team: Chitti | Version: 1.0.0")
    print("=" * 60)


def show_system_status():
    """Show current system configuration and status"""
    print("\n📋 SYSTEM STATUS")
    print("-" * 40)
    
    # Current mode
    current_mode = mode_selector.mode
    print(f"Current Mode: {current_mode.value.upper()}")
    
    # Component choices
    from src.config.mode_selector import ComponentType
    llm_choice = mode_selector.get_component_choice(ComponentType.LLM)
    embeddings_choice = mode_selector.get_component_choice(ComponentType.EMBEDDINGS)
    vectorstore_choice = mode_selector.get_component_choice(ComponentType.VECTORSTORE)
    database_choice = mode_selector.get_component_choice(ComponentType.DATABASE)
    
    print(f"LLM: {llm_choice}")
    print(f"Embeddings: {embeddings_choice}")
    print(f"Vector Store: {vectorstore_choice}")
    print(f"Database: {database_choice}")
    
    print("\n🔍 COMPONENT AVAILABILITY")
    print("-" * 40)
    
    # LLM availability
    llm_info = get_available_llms()
    print("LLMs:")
    for llm_name, info in llm_info.items():
        status = "✅" if info["available"] else "❌"
        print(f"  {status} {info['name']} ({info['type']})")
    
    # Embeddings availability
    embeddings_info = get_available_embeddings()
    print("\nEmbeddings:")
    for emb_name, info in embeddings_info.items():
        status = "✅" if info["available"] else "❌"
        print(f"  {status} {info['name']} ({info['type']})")


def test_llm():
    """Test LLM functionality"""
    print("\n🤖 TESTING LLM")
    print("-" * 40)
    
    try:
        llm = get_llm()
        print(f"✅ LLM loaded: {llm.get_model_info()['model']}")
        
        # Test generation
        test_prompt = "What is the capital of France?"
        print(f"Test prompt: {test_prompt}")
        
        response = llm.generate(test_prompt, max_tokens=50)
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False


def test_document_processing():
    """Test document processing functionality"""
    print("\n📄 TESTING DOCUMENT PROCESSING")
    print("-" * 40)
    
    try:
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        print(f"✅ Document processor initialized")
        print(f"Supported formats: {', '.join(processor.get_supported_formats())}")
        
        # Test processing local documents
        docs_path = "src/database/sample_docs"
        chunks = processor.process_directory(docs_path)
        
        if chunks:
            print(f"✅ Processed {len(chunks)} chunks from documents")
            
            # Show stats
            stats = processor.get_processing_stats(chunks)
            print(f"📊 Processing Statistics:")
            print(f"  - Total chunks: {stats['total_chunks']}")
            print(f"  - Unique sources: {stats['unique_sources']}")
            print(f"  - File types: {stats['file_types']}")
            print(f"  - Avg chunk size: {stats['average_chunk_size']:.0f} chars")
            
            # Show sample chunk
            sample_chunk = chunks[0]
            print(f"\n📝 Sample chunk from {sample_chunk.metadata['file_name']}:")
            print(f"Content preview: {sample_chunk.content[:200]}...")
            
            return True, chunks
        else:
            print("❌ No documents were processed")
            return False, []
            
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False, []


def test_embeddings_with_documents(chunks):
    """Test embeddings with actual document chunks"""
    print("\n📊 TESTING EMBEDDINGS WITH DOCUMENTS")
    print("-" * 40)
    
    if not chunks:
        print("⚠️  No document chunks available for embedding test")
        return False
    
    try:
        embeddings = get_embeddings()
        
        # Test with first few chunks
        test_chunks = chunks[:3]
        print(f"Testing embeddings with {len(test_chunks)} document chunks...")
        
        # Embed chunks
        texts = [chunk.content for chunk in test_chunks]
        chunk_embeddings = embeddings.embed_texts(texts)
        
        print(f"✅ Generated embeddings shape: {chunk_embeddings.shape}")
        
        # Test similarity search
        query = "What is the education system structure in India?"
        query_embedding = embeddings.embed_query(query)
        
        similar_chunks = embeddings.find_most_similar(
            query_embedding, chunk_embeddings, top_k=2
        )
        
        print(f"\n🔍 Query: {query}")
        print(f"Top similar chunks:")
        for result in similar_chunks:
            idx = result['index']
            similarity = result['similarity']
            chunk = test_chunks[idx]
            print(f"  - Similarity: {similarity:.3f} | Source: {chunk.metadata['file_name']}")
            print(f"    Preview: {chunk.content[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings with documents test failed: {e}")
        return False


def interactive_mode_selection():
    """Interactive mode selection"""
    print("\n🔧 MODE SELECTION")
    print("-" * 40)
    
    modes = {
        "1": RAGMode.ONLINE,
        "2": RAGMode.LOCAL,
        "3": RAGMode.HYBRID
    }
    
    print("Available modes:")
    print("1. Online Mode (Azure OpenAI + OpenAI Embeddings + Qdrant + MongoDB Atlas)")
    print("2. Local Mode (Phi-3 + Local Embeddings + FAISS + Local MongoDB)")
    print("3. Hybrid Mode (Mix and match components)")
    print("4. Use environment variables (current)")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice in modes:
        selected_mode = modes[choice]
        
        # Set environment variable
        os.environ["RAG_MODE"] = selected_mode.value
        
        # Reinitialize mode selector to pick up the change
        from src.config.mode_selector import ModeSelector
        global mode_selector
        mode_selector = ModeSelector()
        
        print(f"✅ Mode set to: {selected_mode.value.upper()}")
        
    elif choice == "4":
        print("✅ Using current environment configuration")
    
    else:
        print("❌ Invalid choice. Using current configuration.")


def run_interactive_test():
    """Run interactive test session"""
    print("\n💬 INTERACTIVE TEST")
    print("-" * 40)
    print("Type 'quit' to exit")
    
    try:
        llm = get_llm()
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                response = llm.generate(user_input, max_tokens=200)
                print(f"Assistant: {response}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Could not start interactive test: {e}")


def main():
    """Main application function"""
    print_banner()
    
    # Show system status
    show_system_status()
    
    # Interactive mode selection
    interactive_mode_selection()
    
    print("\n🧪 RUNNING TESTS")
    print("=" * 40)
    
    # Test components
    llm_working = test_llm()
    
    # Test document processing
    doc_working, document_chunks = test_document_processing()
    
    # Test embeddings with documents
    embeddings_working = test_embeddings_with_documents(document_chunks)
    
    print("\n📊 TEST RESULTS")
    print("-" * 40)
    print(f"LLM: {'✅ Working' if llm_working else '❌ Failed'}")
    print(f"Document Processing: {'✅ Working' if doc_working else '❌ Failed'}")
    print(f"Embeddings: {'✅ Working' if embeddings_working else '❌ Failed'}")
    
    # Show system readiness
    if all([llm_working, doc_working, embeddings_working]):
        print("\n🎉 ALL SYSTEMS READY!")
        print("Your RAG system is fully functional with:")
        print(f"  - {len(document_chunks)} processed document chunks")
        print("  - Working LLM for text generation")  
        print("  - Working embeddings for similarity search")
    else:
        print("\n⚠️  SOME COMPONENTS NEED ATTENTION")
        if not llm_working:
            print("  - Check LLM configuration and dependencies")
        if not doc_working:
            print("  - Check document processing setup")
        if not embeddings_working:
            print("  - Check embeddings configuration and dependencies")
    
    # Interactive test if components work
    if llm_working:
        run_interactive = input("\nRun interactive test? (y/n): ").strip().lower()
        if run_interactive == 'y':
            run_interactive_test()
    
    print("\n✅ Application completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user.")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Application error: {e}")
        sys.exit(1)