
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from app.endpoints.research_sources import research
from app.shared import ResearchRequest, ResearchResult, User

class TestMetaSearch(unittest.IsolatedAsyncioTestCase):

    async def test_meta_search_orchestration(self):
        # Mock dependencies
        mock_request = MagicMock()
        mock_db = MagicMock()
        mock_user = User(id=1, email="test@example.com", is_active=True)

        # Mock sub-search functions
        with patch("app.endpoints.research_sources.research_with_grok", new_callable=AsyncMock) as mock_grok, \
             patch("app.endpoints.research_sources.research_with_gemini", new_callable=AsyncMock) as mock_gemini, \
             patch("app.endpoints.research_sources.search_asylnet_with_provisions", new_callable=AsyncMock) as mock_asylnet, \
             patch("app.endpoints.research.meta.aggregate_search_results", new_callable=AsyncMock) as mock_aggregator:

            # Setup mock returns
            mock_grok.return_value = ResearchResult(query="q", summary="Grok Sum", sources=[{"title": "Grok Source", "url": "http://grok.com"}])
            mock_gemini.return_value = ResearchResult(query="q", summary="Gemini Sum", sources=[{"title": "Gemini Source", "url": "http://google.com"}])
            
            # Asylnet returns a dict
            mock_asylnet.return_value = {
                "keywords": ["kw1"], 
                "asylnet_sources": [{"title": "Asyl Source", "url": "http://asyl.net"}],
                "legal_sources": [{"title": "Law Source", "url": "http://gesetze.de"}]
            }

            mock_aggregator.return_value = ResearchResult(
                query="q", 
                summary="Meta Summary", 
                sources=[
                    {"title": "Grok Source", "url": "http://grok.com"},
                    {"title": "Asyl Source", "url": "http://asyl.net"}
                ],
                suggestions=["kw1"]
            )

            # Define request
            req = ResearchRequest(
                query="Test Query",
                search_engine="meta",
                asylnet_keywords="ManualKeyword"
            )

            # Execute
            result = await research(mock_request, req, mock_db, mock_user)

            # Assertions
            self.assertEqual(result.summary, "Meta Summary")
            self.assertEqual(len(result.sources), 2)
            
            # Verify calls
            mock_grok.assert_called_once()
            mock_gemini.assert_called_once()
            
            # Critical: Verify manual keywords passed to asylnet
            mock_asylnet.assert_called_once()
            call_args = mock_asylnet.call_args
            self.assertEqual(call_args.kwargs.get("manual_keywords"), "ManualKeyword")
            
            # Verify aggregation called
            mock_aggregator.assert_called_once()
            
            print("Meta-search orchestration test passed!")

if __name__ == '__main__':
    unittest.main()
