from unstructured.partition.auto import partition
import os
import tempfile # Added for temporary file handling

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

# New function to handle uploaded files
def parse_uploaded_document(file: bytes, filename: str):
    """
    Saves an uploaded file temporarily and then parses it.

    Args:
        file: The file content as bytes.
        filename: The original name of the file.

    Returns:
        A list of Element objects or None if an error occurs.
    """
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name
        
        elements = parse_document(tmp_file_path)
        return elements
    except Exception as e:
        print(f"Error processing uploaded document: {e}")
        return None
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

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