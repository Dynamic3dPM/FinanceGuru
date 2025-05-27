#!/bin/bash

# Installation script for FinanceGuru with Hugging Face and RAG support

echo "Installing FinanceGuru dependencies with Hugging Face Transformers and RAG support..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment detected. Consider creating one with:"
    echo "python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit 1

echo "Installing Python dependencies..."
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "Creating necessary directories..."
mkdir -p models chroma_db

echo ""
echo "Initializing databases..."
python -c "
try:
    from app.services.rag_service import rag_service
    from app.services.hf_llm_service import hf_llm_service
    print('RAG service initialized successfully')
    print('Hugging Face LLM service initialized successfully')
except Exception as e:
    print(f'Warning: Could not initialize services: {e}')
    print('This is normal on first install - services will initialize when first used')
"

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Configure your environment variables in backend/.env (optional)"
echo "2. Start the FastAPI server: cd backend && uvicorn app.main:app --reload"
echo "3. The first time you use a Hugging Face model, it will be downloaded automatically"
echo ""
echo "Key features now available:"
echo "- Hugging Face Transformers for local LLM inference"
echo "- RAG (Retrieval-Augmented Generation) with ChromaDB"
echo "- Automatic document embedding and retrieval"
echo "- Enhanced financial analysis with context from previous documents"
echo ""
echo "Default models:"
echo "- LLM: microsoft/DialoGPT-medium (can be changed in config)"
echo "- Embeddings: sentence-transformers/all-MiniLM-L6-v2"
