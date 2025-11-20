import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.append("/var/opt/docker/rechtmaschine/app")

# Mock environment variables before importing shared
os.environ["GOOGLE_API_KEY"] = "fake_key"
os.environ["OPENAI_API_KEY"] = "fake_key"
os.environ["ANTHROPIC_API_KEY"] = "fake_key"
os.environ["XAI_API_KEY"] = "fake_key"

from endpoints.generation import _upload_documents_to_gemini, _generate_with_gemini
from google.genai import types

class TestGeminiIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        
    @patch("endpoints.generation.get_document_for_upload")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=b"fake_content")
    @patch("os.unlink")
    def test_upload_documents(self, mock_unlink, mock_open, mock_get_doc):
        # Setup
        documents = [
            {"filename": "test.pdf", "file_path": "/tmp/test.pdf"},
            {"filename": "test.txt", "extracted_text_path": "/tmp/test.txt"}
        ]
        
        # Mock get_document_for_upload to return fake paths
        mock_get_doc.side_effect = [
            ("/tmp/test.pdf", "application/pdf", False),
            ("/tmp/test.txt", "text/plain", True)
        ]
        
        # Mock client.files.upload
        mock_upload_file = MagicMock()
        mock_upload_file.uri = "https://gemini.google.com/file/123"
        mock_upload_file.name = "files/123"
        mock_upload_file.state.name = "ACTIVE"
        self.mock_client.files.upload.return_value = mock_upload_file
        self.mock_client.files.get.return_value = mock_upload_file

        # Execute
        files = _upload_documents_to_gemini(self.mock_client, documents)
        
        # Verify
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0].name, "files/123")
        self.assertEqual(files[0].state.name, "ACTIVE")
        
        # Check that unlink was called for the second file (needs_cleanup=True)
        mock_unlink.assert_called_with("/tmp/test.txt")

    def test_generate_content(self):
        # Setup
        system_prompt = "You are a lawyer."
        user_prompt = "Draft a document."
        mock_file = MagicMock()
        mock_file.name = "files/123"
        files = [mock_file]
        
        mock_response = MagicMock()
        mock_response.text = "Generated draft."
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        
        self.mock_client.models.generate_content.return_value = mock_response
        
        # Execute
        result = _generate_with_gemini(
            self.mock_client,
            system_prompt,
            user_prompt,
            files,
            model="gemini-3-pro-preview"
        )
        
        # Verify
        self.assertEqual(result, "Generated draft.")
        self.mock_client.models.generate_content.assert_called_once()
        
        # Verify arguments
        call_args = self.mock_client.models.generate_content.call_args
        self.assertEqual(call_args.kwargs["model"], "gemini-3-pro-preview")
        self.assertEqual(call_args.kwargs["contents"], [mock_file, user_prompt])
        self.assertEqual(call_args.kwargs["config"].system_instruction, system_prompt)

if __name__ == "__main__":
    unittest.main()
