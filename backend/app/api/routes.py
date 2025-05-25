from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List # Keep if needed for other response models

# Placeholder for future service integrations
from app.services.document_parser import parse_uploaded_document # Updated import
from app.services.llm_service import analyze_text_with_llm
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
async def upload_financial_statement(file: UploadFile = File(...)):
    """
    Uploads and parses a financial statement.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Basic file validation (example)
    # You might want to expand this or make it more robust
    # allowed_content_types = ["application/pdf", "image/jpeg", "image/png", "text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    # if file.content_type not in allowed_content_types:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Invalid file type: {file.content_type}. Allowed types: {', '.join(allowed_content_types)}"
    #     )

    try:
        content = await file.read()
        # Use the new function from document_parser
        parsed_elements = parse_uploaded_document(content, file.filename)

        if not parsed_elements:
            raise HTTPException(status_code=500, detail="Failed to parse document")

        # For now, let's just return the number of elements found.
        # You can adapt this to return the actual content or a summary.
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "message": f"Successfully parsed {file.filename}. Found {len(parsed_elements)} elements.",
            # "elements": [str(el) for el in parsed_elements] # Optionally return elements
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
async def analyze_data(request: schemas.AnalysisRequest):
    analysis_result = await analyze_text_with_llm(request.text_content)
    return analysis_result
    # return {"summary": "Analysis placeholder", "suggestions": ["Suggestion 1"]}

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