# Team Chitti RAG - Local Setup Guide (TinyLlama + FAISS)

## Overview
This guide will help you set up **Team Chitti RAG** in **Local Mode** using:
- **LLM**: TinyLlama (1.1B parameters - lightweight & fast)
- **Embeddings**: Sentence Transformers (local, offline)
- **Vector Store**: FAISS (local vector similarity search)
- **Database**: MongoDB Local instance

## System Requirements

### Minimum Requirements
- **RAM**: 4-6 GB (for TinyLlama + embeddings)
- **Storage**: 10 GB free space (for models + data)
- **Python**: 3.10+
- **Processor**: Any CPU (works on Intel/AMD/Apple)

### Recommended Requirements
- **RAM**: 8-16 GB
- **GPU**: Optional but helpful (NVIDIA CUDA or Apple Metal)
- **Storage**: 20+ GB free space

## Installation Steps

### 1. Install Required Packages

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install manually
pip install torch transformers accelerate
pip install sentence-transformers
pip install faiss-cpu  # Use faiss-gpu if you have NVIDIA GPU
pip install pymongo motor
pip install fastapi uvicorn
pip install pydantic python-dotenv
```

### 2. Setup Environment

The `.env` file is already configured for local mode:

```bash
# Verify RAG_MODE is set to local
cat .env | grep RAG_MODE

# Should output:
# RAG_MODE=local
```

**Key configuration values:**
```env
RAG_MODE=local                                                    # Local mode
TINYLLAMA_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0          # LLM model
TINYLLAMA_DEVICE=auto                                             # auto, cuda, or cpu
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2     # Embeddings
FAISS_INDEX_PATH=./data/vector_store/faiss_index                 # Vector store
MONGODB_LOCAL_URI=mongodb://localhost:27017                       # Database
```

### 3. Install and Start MongoDB

```bash
# macOS (using Homebrew)
brew install mongodb-community
brew services start mongodb-community

# Ubuntu/Debian
sudo apt-get install -y mongodb
sudo systemctl start mongodb

# Windows
# Download from: https://www.mongodb.com/try/download/community
# Run the installer and follow the wizard

# Verify MongoDB is running
mongo --version && mongosh --eval "db.version()"
```

### 4. Run Setup Script

```bash
# Run the setup script to test all components
python scripts/local_setup.py

# To skip component tests (just create directories)
python scripts/local_setup.py --skip-tests

# To skip TinyLlama/embeddings download
python scripts/local_setup.py --skip-llm --skip-embeddings
```

The setup script will:
- ✅ Check all dependencies
- ✅ Create necessary directories
- ✅ Download TinyLlama model
- ✅ Download Sentence Transformers model
- ✅ Test FAISS vector store
- ✅ Test MongoDB connection
- ✅ Validate configuration

## Quick Start

### 1. Process Documents

```bash
# Process sample education documents
python -m src.database.document_processor ./src/database/sample_docs

# This will:
# - Read all documents from the folder
# - Split them into chunks
# - Store metadata
```

### 2. Create Embeddings and Index

```python
from src.embeddings.embeddings_factory import get_embeddings
from src.vectorstore.vectorstore_factory import get_vectorstore
from src.database.document_processor import DocumentProcessor
import numpy as np

# Load components
embeddings = get_embeddings()
vectorstore = get_vectorstore()
processor = DocumentProcessor()

# Process documents
chunks = processor.process_directory('./src/database/sample_docs')

# Create embeddings and add to index
for chunk in chunks:
    embedding = embeddings.embed_text(chunk.content)
    vectorstore.add_vectors([embedding], [{
        'chunk_id': chunk.chunk_id,
        'content': chunk.content,
        'source': chunk.source_file
    }])

# Save the index
vectorstore.save_index()
```

### 3. Query the System

```python
from src.llm.llm_factory import get_llm
from src.embeddings.embeddings_factory import get_embeddings
from src.vectorstore.vectorstore_factory import get_vectorstore

# Load components
llm = get_llm()
embeddings = get_embeddings()
vectorstore = get_vectorstore()

# Query
question = "What is the National Education Policy 2020?"

# Get relevant documents
query_embedding = embeddings.embed_text(question)
results = vectorstore.search_with_metadata(query_embedding, k=5)

# Build context from results
context = "\n".join([r['metadata']['content'] for r in results])

# Generate answer
prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {question}

Answer:"""

answer = llm.generate(prompt)
print(answer)
```

### 4. Run the Application

```bash
# Start the FastAPI server
python main.py

# Or explicitly specify development mode
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Key Features

### 🤖 TinyLlama LLM
- **Size**: 1.1B parameters (~2.5 GB disk)
- **Speed**: ⚡ Very fast inference (1-2 seconds per response)
- **Quality**: Good for educational Q&A
- **Offline**: ✅ Fully offline capability
- **Device**: Works on CPU or GPU

### 🔗 Sentence Transformers Embeddings
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Speed**: ⚡ Fast embedding generation
- **Quality**: ✅ Good for semantic search
- **Size**: ~80 MB

### 🗂️ FAISS Vector Store
- **Speed**: ⚡ Ultra-fast similarity search
- **Scalability**: Can handle millions of vectors
- **Index Types**: Flat (best quality), IVF (balanced)
- **Persistence**: Saves to disk

### 💾 MongoDB Local
- **Database**: Full-featured document database
- **Storage**: All documents and metadata
- **Query**: Complex queries on document data
- **Backup**: Easy backup and recovery

## Configuration Files

### `.env` - Environment Configuration
Located at the root of the project. Configure:
- Model paths and devices
- Embedding models
- Vector store settings
- Database connection
- API settings

### `src/config/local_config.py`
Pydantic models for local configuration. Defines all environment variables for local mode.

### `src/config/mode_selector.py`
Determines which components to use based on the selected mode (online/local/hybrid).

## Performance Tips

### ✅ Best Practices
1. **Use CPU for embeddings** - Fast and doesn't need GPU
2. **Use FAISS Flat index** - Better quality for small/medium datasets
3. **Batch embeddings** - Process multiple documents together
4. **Cache embeddings** - Save computed embeddings to avoid recalculation
5. **Use appropriate chunk sizes** - Larger chunks for complex docs, smaller for Q&A

### 🚀 Optimization Tips
1. **Use GPU if available**:
   ```env
   TINYLLAMA_DEVICE=cuda
   LOCAL_EMBEDDING_DEVICE=cuda
   ```

2. **Switch to HNSW index for large datasets**:
   ```env
   FAISS_INDEX_TYPE=HNSW
   ```

3. **Use batch processing**:
   ```env
   BATCH_SIZE=64
   MAX_WORKERS=8
   ```

## Troubleshooting

### Issue: Out of Memory
**Solution:**
1. Reduce chunk size: `CHUNK_SIZE=500`
2. Use CPU only: `TINYLLAMA_DEVICE=cpu`
3. Process documents in smaller batches

### Issue: MongoDB Connection Failed
**Solution:**
```bash
# Check if MongoDB is running
mongosh

# Or start it
sudo systemctl start mongodb  # Linux
brew services start mongodb-community  # macOS
```

### Issue: Slow Inference
**Solution:**
1. Use GPU: `TINYLLAMA_DEVICE=cuda`
2. Reduce MAX_TOKENS: `MAX_TOKENS=500`
3. Use smaller batch sizes

### Issue: Models Not Downloading
**Solution:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python scripts/local_setup.py --skip-tests
```

## Directory Structure

```
team-chitti-rag/
├── models/                          # Downloaded models
│   ├── tinyllama/                  # TinyLlama model
│   └── embeddings/                 # Sentence Transformers
├── data/                            # Data files
│   ├── vector_store/               # FAISS indices
│   ├── documents/                  # Processed documents
│   └── embeddings/                 # Cached embeddings
├── logs/                            # Application logs
├── src/
│   ├── config/                     # Configuration
│   │   ├── mode_selector.py        # Mode selection
│   │   ├── local_config.py         # Local settings
│   │   └── hybrid_config.py        # Unified config
│   ├── llm/                        # LLM implementations
│   │   ├── tinyllama_local_llm.py
│   │   └── llm_factory.py
│   ├── embeddings/                 # Embedding implementations
│   │   ├── local_embeddings.py
│   │   └── embeddings_factory.py
│   ├── vectorstore/                # Vector store implementations
│   │   ├── faiss_vectorstore.py
│   │   └── vectorstore_factory.py
│   ├── database/                   # Database
│   │   └── document_processor.py
│   └── api/                        # FastAPI endpoints
├── scripts/                        # Utility scripts
│   └── local_setup.py              # Setup script
├── main.py                         # Main application
├── .env                            # Environment variables
└── requirements.txt                # Dependencies
```

## Next Steps

1. ✅ **Set up infrastructure** (you are here)
2. 📚 **Process your documents**
3. 🔍 **Create embeddings and build index**
4. 💬 **Test Q&A with real queries**
5. 🚀 **Deploy to production**

## Additional Resources

- [TinyLlama GitHub](https://github.com/jzhang38/TinyLlama)
- [Sentence Transformers Docs](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [MongoDB Docs](https://docs.mongodb.com/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `./logs/`
3. Run the setup script with verbose output: `python scripts/local_setup.py`

---

**Happy RAGing! 🚀**
