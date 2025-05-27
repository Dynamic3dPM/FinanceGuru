from unstructured.partition.auto import partition
import os
import tempfile # Added for temporary file handling
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.services.rag_service import rag_service
import re
from dateutil import parser as date_parser
from datetime import datetime, date

def parse_document(file_path: str):
    """
    Parses a document using the unstructured library.
    
    Args:
        file_path: The path to the document to parse.
        
    Returns:
        A list of Element objects representing the partitioned document.
    """
    try:
        elements = partition(filename=file_path)
        return elements
    except Exception as e:
        print(f"Error parsing document: {e}")
        return None

def extract_text_from_elements(elements) -> str:
    """
    Extract clean text from unstructured elements.
    
    Args:
        elements: List of unstructured elements
        
    Returns:
        Concatenated text content
    """
    if not elements:
        return ""
    
    text_parts = []
    for element in elements:
        if hasattr(element, 'text') and element.text:
            text_parts.append(element.text.strip())
    
    return "\n".join(text_parts)

def parse_uploaded_document(file: bytes, filename: str, add_to_rag: bool = True) -> Dict[str, Any]:
    """
    Saves an uploaded file temporarily, parses it, and optionally adds to RAG.

    Args:
        file: The file content as bytes.
        filename: The original name of the file.
        add_to_rag: Whether to add the document to the RAG system.

    Returns:
        Dictionary containing parsing results and document info.
    """
    document_id = str(uuid.uuid4())
    tmp_file_path = None
    
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name
        
        # Parse the document
        elements = parse_document(tmp_file_path)
        
        if not elements:
            return {
                "success": False,
                "error": "Failed to parse document",
                "document_id": document_id,
                "elements": None,
                "text_content": "",
                "added_to_rag": False
            }
        
        # Extract text content
        text_content = extract_text_from_elements(elements)
        
        # Extract date information from the document
        date_info = extract_dates_from_text(text_content)
        
        # Add to RAG system if requested
        added_to_rag = False
        if add_to_rag and text_content.strip():
            try:
                # Flatten extracted dates for ChromaDB metadata (it only accepts simple types)
                metadata = {
                    "filename": filename,
                    "uploaded_at": datetime.now().isoformat(),
                    "file_size": len(file),
                    "content_type": _get_content_type(filename),
                    # Add date information to metadata
                    "statement_date": date_info.get("statement_date"),
                    "month_year": date_info.get("month_year"),
                    "statement_period": date_info.get("statement_period"),
                    # Flatten extracted dates into simple metadata fields
                    "extracted_dates_count": len(date_info.get("extracted_dates", [])),
                }
                
                # Add the first few extracted dates as individual fields (flattened)
                extracted_dates = date_info.get("extracted_dates", [])
                for i, date_item in enumerate(extracted_dates[:3]):  # Limit to first 3 dates
                    metadata[f"date_{i}_raw"] = date_item.get("raw_text", "")
                    metadata[f"date_{i}_parsed"] = date_item.get("parsed_date", "")
                    metadata[f"date_{i}_month"] = date_item.get("month", 0)
                    metadata[f"date_{i}_year"] = date_item.get("year", 0)
                    metadata[f"date_{i}_month_name"] = date_item.get("month_name", "")
                
                added_to_rag = rag_service.add_document(
                    document_id=document_id,
                    filename=filename,
                    content=text_content,
                    content_type=metadata["content_type"],
                    metadata=metadata
                )
                
                if added_to_rag:
                    print(f"Successfully added document '{filename}' to RAG system with ID: {document_id}")
                else:
                    print(f"Failed to add document '{filename}' to RAG system")
                    
            except Exception as rag_error:
                print(f"Error adding document to RAG: {rag_error}")
                added_to_rag = False
        
        return {
            "success": True,
            "document_id": document_id,
            "elements": elements,
            "text_content": text_content,
            "added_to_rag": added_to_rag,
            "filename": filename,
            "metadata": {
                "uploaded_at": datetime.now().isoformat(),
                "file_size": len(file),
                "content_type": _get_content_type(filename),
                "element_count": len(elements)
            }
        }
        
    except Exception as e:
        print(f"Error processing uploaded document: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
            "elements": None,
            "text_content": "",
            "added_to_rag": False
        }
    finally:
        # Clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def _get_content_type(filename: str) -> str:
    """Determine content type based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.html': 'text/html'
    }
    return content_types.get(ext, 'application/octet-stream')

def extract_dates_from_text(text: str) -> Dict[str, Any]:
    """
    Extract date information from financial document text.
    
    Args:
        text: The document text content
        
    Returns:
        Dictionary containing extracted date information
    """
    date_info = {
        "statement_period": None,
        "statement_date": None,
        "month_year": None,
        "extracted_dates": []
    }
    
    if not text:
        return date_info
    
    # Common financial statement date patterns
    date_patterns = [
        r'statement\s+(?:period|date):\s*([^,\n]+)',
        r'(?:for\s+the\s+period|period\s+ending):\s*([^,\n]+)',
        r'(?:statement\s+date|as\s+of):\s*([^,\n]+)',
        r'(?:billing\s+period|cycle):\s*([^,\n]+)',
        r'(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})'
    ]
    
    text_lower = text.lower()
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            date_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
            try:
                parsed_date = date_parser.parse(date_str.strip())
                date_info["extracted_dates"].append({
                    "raw_text": date_str.strip(),
                    "parsed_date": parsed_date.isoformat(),
                    "month": parsed_date.month,
                    "year": parsed_date.year,
                    "month_name": parsed_date.strftime("%B")
                })
                
                # Set primary dates if not already set
                if not date_info["statement_date"]:
                    date_info["statement_date"] = parsed_date.isoformat()
                    date_info["month_year"] = f"{parsed_date.strftime('%B')} {parsed_date.year}"
                    
            except (ValueError, TypeError):
                continue
    
    # If no specific dates found, try to extract month/year from filename or content
    if not date_info["extracted_dates"]:
        month_year_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b'
        month_matches = re.finditer(month_year_pattern, text_lower, re.IGNORECASE)
        for match in month_matches:
            month_name = match.group(1).capitalize()
            year = int(match.group(2))
            try:
                # Create a date for the first of the month
                parsed_date = datetime(year, datetime.strptime(month_name, "%B").month, 1)
                date_info["extracted_dates"].append({
                    "raw_text": f"{month_name} {year}",
                    "parsed_date": parsed_date.isoformat(),
                    "month": parsed_date.month,
                    "year": parsed_date.year,
                    "month_name": month_name
                })
                
                if not date_info["statement_date"]:
                    date_info["statement_date"] = parsed_date.isoformat()
                    date_info["month_year"] = f"{month_name} {year}"
                    
            except ValueError:
                continue
    
    # Set statement period if we have dates
    if date_info["extracted_dates"]:
        sorted_dates = sorted(date_info["extracted_dates"], key=lambda x: x["parsed_date"])
        if len(sorted_dates) >= 2:
            date_info["statement_period"] = f"{sorted_dates[0]['month_name']} {sorted_dates[0]['year']} - {sorted_dates[-1]['month_name']} {sorted_dates[-1]['year']}"
        else:
            date_info["statement_period"] = date_info["month_year"]
    
    return date_info

# Legacy function for backward compatibility
def parse_uploaded_document_legacy(file: bytes, filename: str):
    """
    Legacy function that maintains the original API.
    
    Args:
        file: The file content as bytes.
        filename: The original name of the file.

    Returns:
        A list of Element objects or None if an error occurs.
    """
    result = parse_uploaded_document(file, filename, add_to_rag=False)
    return result.get("elements") if result.get("success") else None

if __name__ == '__main__':
    # Example usage:
    # Create a dummy file for testing
    dummy_file_path = "dummy_document.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a test document.\n\nIt has multiple paragraphs.")
    
    parsed_elements = parse_document(dummy_file_path)
    
    if parsed_elements:
        for element in parsed_elements:
            print(element)
            
    # Clean up the dummy file
    os.remove(dummy_file_path)