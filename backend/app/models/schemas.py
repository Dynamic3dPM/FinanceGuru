from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date as datetime_date # Alias date to avoid conflict

class FileUploadResponse(BaseModel):
    filename: str
    content_type: str
    message: str
    document_id: Optional[str] = None
    added_to_rag: Optional[bool] = None

class AnalysisRequest(BaseModel):
    text_content: str # Or structured data extracted from the document
    # user_id: Optional[str] = None # If you add user accounts

class Transaction(BaseModel):
    date: Optional[datetime_date] = None # Use the aliased date
    description: str
    amount: float
    category: Optional[str] = None

class FinancialAnalysis(BaseModel):
    summary: str
    suggestions: List[str]
    transactions_identified: Optional[List[Transaction]] = None
    spoken_response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # raw_llm_response: Optional[Dict[Any, Any]] = None # For debugging

class TTSRequest(BaseModel):
    text: str
    # language_code: str = "en-US"
    # voice_name: Optional[str] = None # e.g., "en-US-Wavenet-D"

class TTSResponse(BaseModel):
    audio_content_url: Optional[str] = None # If you save it and provide a URL
    message: str
    # audio_base64: Optional[str] = None # If sending audio data directly

# RAG-related schemas
class RAGDocument(BaseModel):
    id: str
    filename: str
    content_type: str
    uploaded_at: str
    chunk_count: int
    metadata: Dict[str, Any]

class RAGSearchResult(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

class RAGSearchResponse(BaseModel):
    success: bool
    query: str
    results: List[RAGSearchResult]
    count: int

class DocumentListResponse(BaseModel):
    success: bool
    documents: List[RAGDocument]
    count: int

class DocumentDeleteResponse(BaseModel):
    success: bool
    message: str

class AddTextToRAGRequest(BaseModel):
    text_content: str
    document_name: str
    metadata: Optional[Dict[str, Any]] = None

class AddTextToRAGResponse(BaseModel):
    success: bool
    document_id: str
    message: str

# Enhanced analysis request with RAG options
class EnhancedAnalysisRequest(BaseModel):
    text_content: str
    use_rag: bool = True
    use_hf: bool = True  # Use Hugging Face by default
    context_query: Optional[str] = None  # Custom query for RAG context