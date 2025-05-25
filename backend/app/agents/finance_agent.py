from app.services.document_parser import parse_uploaded_document
from app.services.llm_service import analyze_text_with_llm
from app.services.tts_service import generate_speech_from_text

class FinanceAIAgent:
    async def process_statement(self, file_content: bytes, filename: str, tts: bool = False):
        parsed_elements = parse_uploaded_document(file_content, filename)
        if not parsed_elements:
            raise ValueError("Failed to parse document")
        text_content = "\n".join(str(el) for el in parsed_elements)
        analysis = await analyze_text_with_llm(text_content)
        if tts and hasattr(analysis, 'spoken_response') and analysis.spoken_response:
            tts_result = generate_speech_from_text(analysis.spoken_response)
            analysis.spoken_audio_url = tts_result.get('audio_content_url')
        return analysis

    async def process_text(self, text_content: str, tts: bool = False):
        analysis = await analyze_text_with_llm(text_content)
        if tts and hasattr(analysis, 'spoken_response') and analysis.spoken_response:
            tts_result = generate_speech_from_text(analysis.spoken_response)
            analysis.spoken_audio_url = tts_result.get('audio_content_url')
        return analysis
