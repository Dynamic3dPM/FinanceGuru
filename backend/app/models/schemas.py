from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date as datetime_date # Alias date to avoid conflict

class FileUploadResponse(BaseModel):
    filename: str
    content_type: str
    message: str
    # You might add a unique ID for the processed document later
    # document_id: Optional[str] = None

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
    # raw_llm_response: Optional[Dict[Any, Any]] = None # For debugging

class TTSRequest(BaseModel):
    text: str
    # language_code: str = "en-US"
    # voice_name: Optional[str] = None # e.g., "en-US-Wavenet-D"

class TTSResponse(BaseModel):
    audio_content_url: Optional[str] = None # If you save it and provide a URL
    message: str
    # audio_base64: Optional[str] = None # If sending audio data directly