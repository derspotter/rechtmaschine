import sys
import os
from unittest.mock import MagicMock, patch

# Add app directory to path
sys.path.append("/var/opt/docker/rechtmaschine/app")

# Mock dependencies
sys.modules["openai"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()

from endpoints.generation import _upload_documents_to_openai

def test_gpt5_upload_logic():
    print("Testing GPT-5.1 file handling logic...")
    
    # Mock data
    documents = [
        {"filename": "text_doc.txt", "extracted_text_path": "mock_text.txt", "file_path": "mock_text.txt"},
        {"filename": "pdf_doc.pdf", "file_path": "mock_pdf.pdf"}
    ]
    
    # Create mock files
    with open("mock_text.txt", "w") as f:
        f.write("This is some text content.")
    with open("mock_pdf.pdf", "wb") as f:
        f.write(b"%PDF-1.4 mock content")
        
    try:
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_client.files.create.return_value.id = "file-123"
        
        # Mock get_document_for_upload to return appropriate types
        with patch("endpoints.generation.get_document_for_upload") as mock_get_doc:
            def side_effect(entry):
                if entry["filename"] == "text_doc.txt":
                    return ("mock_text.txt", "text/plain", False)
                else:
                    return ("mock_pdf.pdf", "application/pdf", False)
            mock_get_doc.side_effect = side_effect
            
            # Run function
            file_blocks = _upload_documents_to_openai(mock_client, documents)
            
            print("File Blocks:", file_blocks)
            
            # Verify text doc was embedded
            text_block = next((b for b in file_blocks if b["type"] == "input_text"), None)
            assert text_block is not None
            assert "DOKUMENT: text_doc.txt" in text_block["text"]
            assert "This is some text content." in text_block["text"]
            print("✅ Text document correctly embedded")
            
            # Verify PDF was uploaded
            file_block = next((b for b in file_blocks if b["type"] == "input_file"), None)
            assert file_block is not None
            assert file_block["file_id"] == "file-123"
            print("✅ PDF document correctly uploaded")
            
    finally:
        # Cleanup
        if os.path.exists("mock_text.txt"): os.remove("mock_text.txt")
        if os.path.exists("mock_pdf.pdf"): os.remove("mock_pdf.pdf")

if __name__ == "__main__":
    try:
        test_gpt5_upload_logic()
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
