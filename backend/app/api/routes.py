from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Optional # Keep if needed for other response models

# Placeholder for future service integrations
from app.services.document_parser import parse_uploaded_document # Updated import
from app.services.llm_service import (
    analyze_text_with_llm, 
    add_document_to_rag, 
    remove_document_from_rag, 
    list_rag_documents,
    search_rag_documents
)
# from app.services.tts_service import generate_speech_from_text

from app.models import schemas # Import your Pydantic models
from app.agents.finance_agent import FinanceAIAgent

router = APIRouter()
agent = FinanceAIAgent()

@router.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "message": "API is healthy"}

@router.post("/upload-statement", response_model=schemas.FileUploadResponse, tags=["Financial Documents"])
async def upload_financial_statement(
    file: UploadFile = File(...),
    add_to_rag: bool = Query(True, description="Whether to add the document to RAG system for future context")
):
    """
    Uploads and parses a financial statement with optional RAG integration.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        content = await file.read()
        # Use the enhanced function from document_parser
        result = parse_uploaded_document(content, file.filename, add_to_rag=add_to_rag)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to parse document"))

        elements = result["elements"]
        rag_status = "Added to RAG system" if result["added_to_rag"] else "Not added to RAG system"
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "message": f"Successfully parsed {file.filename}. Found {len(elements)} elements. {rag_status}",
            "document_id": result["document_id"],
            "added_to_rag": result["added_to_rag"]
        }
    except HTTPException as e:
        # Re-raise HTTPExceptions to be handled by FastAPI
        raise e
    except Exception as e:
        # Catch any other exceptions during processing
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        await file.close()


# Add more endpoints here for analysis, TTS, etc.
# Example:
@router.post("/analyze-data", response_model=schemas.FinancialAnalysis, tags=["Analysis"])
async def analyze_data(
    request: schemas.AnalysisRequest,
    use_rag: bool = Query(True, description="Whether to use RAG for context retrieval"),
    use_hf: bool = Query(True, description="Whether to use Hugging Face instead of Ollama")
):
    """
    Analyze financial data with optional RAG context and choice of LLM backend.
    """
    analysis_result = await analyze_text_with_llm(
        request.text_content, 
        use_rag=use_rag, 
        use_hf=use_hf
    )
    return analysis_result

# @router.post("/generate-tts", response_model=schemas.TTSResponse, tags=["TTS"])
# async def generate_tts(request: schemas.TTSRequest):
#     # audio_url = await generate_speech_from_text(request.text)
#     # return {"audio_content_url": audio_url, "message": "TTS generated"}
#     return {"message": "TTS placeholder"}

@router.post("/agent-analyze", response_model=schemas.FinancialAnalysis, tags=["Analysis"])
async def agent_analyze(file: UploadFile = File(...)):
    content = await file.read()
    try:
        result = await agent.process_statement(content, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

# RAG Management Endpoints
@router.get("/rag/documents", tags=["RAG Management"])
async def get_rag_documents():
    """
    List all documents in the RAG system.
    """
    try:
        documents = list_rag_documents()
        return {
            "success": True,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing RAG documents: {str(e)}")

@router.delete("/rag/documents/{document_id}", tags=["RAG Management"])
async def delete_rag_document(document_id: str):
    """
    Delete a document from the RAG system.
    """
    try:
        success = remove_document_from_rag(document_id)
        if success:
            return {
                "success": True,
                "message": f"Document {document_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.post("/rag/search", tags=["RAG Management"])
async def search_rag_content(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return", ge=1, le=20)
):
    """
    Search for relevant content in the RAG system.
    """
    try:
        results = search_rag_documents(query, top_k)
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching RAG content: {str(e)}")

@router.post("/rag/add-text", tags=["RAG Management"])
async def add_text_to_rag(
    text_content: str = Query(..., description="Text content to add"),
    document_name: str = Query(..., description="Name for the document"),
    metadata: Optional[dict] = None
):
    """
    Add raw text content to the RAG system.
    """
    try:
        import uuid
        document_id = str(uuid.uuid4())
        success = add_document_to_rag(
            document_id=document_id,
            filename=document_name,
            content=text_content,
            content_type="text/plain",
            metadata=metadata or {}
        )
        
        if success:
            return {
                "success": True,
                "document_id": document_id,
                "message": f"Text content added to RAG system as '{document_name}'"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add text to RAG system")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding text to RAG: {str(e)}")

# Enhanced analysis endpoint with file upload and automatic RAG addition
@router.post("/analyze-file", response_model=schemas.FinancialAnalysis, tags=["Analysis"])
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    use_rag: bool = Query(True, description="Whether to use RAG for context retrieval"),
    add_to_rag: bool = Query(True, description="Whether to add this file to RAG system"),
    use_hf: bool = Query(True, description="Whether to use Hugging Face instead of Ollama")
):
    """
    Upload a file, optionally add it to RAG, and analyze it immediately.
    """
    try:
        content = await file.read()
        
        # Parse and optionally add to RAG
        parse_result = parse_uploaded_document(content, file.filename, add_to_rag=add_to_rag)
        
        if not parse_result["success"]:
            raise HTTPException(status_code=500, detail=parse_result.get("error", "Failed to parse document"))
        
        text_content = parse_result["text_content"]
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")
        
        # Analyze the content
        analysis_result = await analyze_text_with_llm(
            text_content, 
            use_rag=use_rag, 
            use_hf=use_hf
        )
        
        # Add metadata about the upload
        analysis_result.metadata = {
            "document_id": parse_result["document_id"],
            "filename": file.filename,
            "added_to_rag": parse_result["added_to_rag"],
            "element_count": len(parse_result["elements"]) if parse_result["elements"] else 0
        }
        
        return analysis_result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")
    finally:
        await file.close()