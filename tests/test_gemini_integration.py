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
        # Setup — _generate_with_gemini uses the chats API: client.chats.create
        # (model/history/config) + chat.send_message([file parts..., prompt]),
        # and returns (text, TokenUsage).
        system_prompt = "You are a lawyer."
        user_prompt = "Draft a document."
        mock_file = MagicMock()
        mock_file.name = "files/123"
        mock_file.uri = "https://gemini.google.com/file/123"
        mock_file.mime_type = "application/pdf"
        files = [mock_file]

        mock_response = MagicMock()
        mock_response.text = "Generated draft."
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.candidates = []

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response
        self.mock_client.chats.create.return_value = mock_chat

        # Execute
        text, usage = _generate_with_gemini(
            self.mock_client,
            system_prompt,
            user_prompt,
            files,
            model="gemini-3.1-pro-preview"
        )

        # Verify
        self.assertEqual(text, "Generated draft.")
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)

        create_kwargs = self.mock_client.chats.create.call_args.kwargs
        self.assertEqual(create_kwargs["model"], "gemini-3.1-pro-preview")
        self.assertEqual(create_kwargs["history"], [])
        self.assertEqual(create_kwargs["config"].system_instruction, system_prompt)

        # With empty history the files travel in the message itself
        (parts,), _ = mock_chat.send_message.call_args
        self.assertEqual(parts[0].file_data.file_uri, mock_file.uri)
        self.assertEqual(parts[0].file_data.mime_type, "application/pdf")
        self.assertEqual(parts[-1], user_prompt)

if __name__ == "__main__":
    unittest.main()
