#!/bin/bash

# Team Chitti RAG - Setup Script
# Automated setup for the modular RAG system

set -e  # Exit on any error

echo "🚀 Team Chitti RAG - Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_header "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python found: $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        REQUIRED_VERSION="3.8"
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible"
        else
            print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_header "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    else
        print_status "Virtual environment already exists"
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Install requirements
install_requirements() {
    print_header "Installing Python requirements..."
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        pip install -r requirements.txt
        print_status "Requirements installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Setup environment file
setup_env() {
    print_header "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_status "Copying .env.example to .env..."
            cp .env.example .env
            print_warning "Please edit .env file with your API keys and configuration"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_status ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_header "Creating necessary directories..."
    
    DIRECTORIES=(
        "data"
        "data/documents"
        "data/vector_store"
        "logs"
        "models"
        "models/phi3"
        "models/embeddings"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
}

# Check optional dependencies
check_optional_deps() {
    print_header "Checking optional dependencies..."
    
    # Check for GPU support
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        print_status "PyTorch is installed"
        python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    else
        print_warning "PyTorch not found. Some local models may not work."
    fi
    
    # Check MongoDB
    if command -v mongod &> /dev/null; then
        print_status "MongoDB is installed"
    else
        print_warning "MongoDB not found. Local database mode will not work."
        print_warning "To install MongoDB, visit: https://www.mongodb.com/docs/manual/installation/"
    fi
}

# Download sample documents
download_sample_docs() {
    print_header "Setting up sample documents..."
    
    SAMPLE_DIR="data/documents/samples"
    mkdir -p "$SAMPLE_DIR"
    
    # Create some sample documents
    cat > "$SAMPLE_DIR/sample1.txt" << EOF
Team Chitti RAG System

This is a sample document for testing the RAG system. 
The Team Chitti RAG system supports three modes:
1. Online Mode: Uses cloud services like Azure OpenAI, OpenAI embeddings, Qdrant, and MongoDB Atlas
2. Local Mode: Uses local models like Phi-3, sentence transformers, FAISS, and local MongoDB
3. Hybrid Mode: Mix and match components from online and local modes

The system is designed to be modular and flexible.
EOF

    cat > "$SAMPLE_DIR/sample2.txt" << EOF
Technical Architecture

The system follows a modular architecture with the following components:
- LLM Factory: Creates appropriate language models
- Embeddings Factory: Creates appropriate embedding models
- Vector Store Factory: Creates appropriate vector databases
- Database Factory: Creates appropriate document databases

Each factory uses the configuration to determine which implementation to use.
EOF

    print_status "Sample documents created in $SAMPLE_DIR"
}

# Test installation
test_installation() {
    print_header "Testing installation..."
    
    print_status "Testing basic imports..."
    python3 -c "
import sys
sys.path.insert(0, 'src')
from src.config.mode_selector import mode_selector
from src.config.hybrid_config import hybrid_config
print('✅ Basic imports successful')
"
    
    print_status "Installation test completed"
}

# Mode selection help
show_mode_help() {
    print_header "Mode Configuration Help"
    
    echo ""
    echo "The system supports three modes:"
    echo ""
    echo "1. 🌐 ONLINE MODE:"
    echo "   - LLM: Azure OpenAI (GPT-4)"
    echo "   - Embeddings: OpenAI (text-embedding-ada-002)"
    echo "   - Vector Store: Qdrant (cloud)"
    echo "   - Database: MongoDB Atlas"
    echo "   - Requires: API keys, internet connection"
    echo ""
    echo "2. 🤖 LOCAL MODE:"
    echo "   - LLM: Phi-3 Mini (3.8B parameters)"
    echo "   - Embeddings: sentence-transformers/all-MiniLM-L6-v2"
    echo "   - Vector Store: FAISS (local files)"
    echo "   - Database: MongoDB (local installation)"
    echo "   - Requires: ~8GB RAM, ~4GB storage, local MongoDB"
    echo ""
    echo "3. 🔧 HYBRID MODE:"
    echo "   - Mix and match any components"
    echo "   - Example: Azure LLM + Local embeddings + FAISS + Local DB"
    echo "   - Requires: Dependencies for selected components"
    echo ""
}

# Main setup function
main() {
    print_status "Starting Team Chitti RAG setup..."
    
    check_python
    create_venv
    install_requirements
    setup_env
    create_directories
    download_sample_docs
    check_optional_deps
    test_installation
    
    echo ""
    print_status "Setup completed successfully! 🎉"
    echo ""
    print_warning "Next steps:"
    echo "1. Edit the .env file with your API keys and configuration"
    echo "2. Choose your preferred mode (online/local/hybrid)"
    echo "3. Run: source venv/bin/activate && python main.py"
    echo ""
    
    show_mode_help
}

# Handle interrupts
trap 'print_error "Setup interrupted"; exit 1' INT

# Run main function
main