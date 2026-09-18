import ast
import asyncio
import os
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx

source_path = Path(os.environ.get('OCR_MANAGER_SOURCE', str(Path(__file__).resolve().parents[1] / 'service_manager.py')))
source = ast.parse(source_path.read_text())
http = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_http_ocr')
endpoint = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'ocr_document')
work = next(n for n in endpoint.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'do_ocr')

class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail

class Tests(unittest.IsolatedAsyncioTestCase):
    def namespace(self):
        return dict(httpx=httpx, HTTPException=HTTPException, os=os, Path=Path,
                    time=time, request_start=time.time(), filename='test.pdf', file_content=b'pdf',
                    request_id='test', RAG_REQUEST_ID_HEADER='X-Request-ID', log=lambda *a: None,
                    _as_float_env=lambda key, default: float(os.environ.get(key, default)),
                    SERVICES={'ocr': {'use_http_service': True, 'url': 'http://unused'}})

    async def test_timeout_configuration(self):
        ns = self.namespace()
        exec(compile(ast.Module(body=[http], type_ignores=[]), '<test>', 'exec'), ns)
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = types.SimpleNamespace(status_code=200, json=lambda: {'full_text': 'ok'})
        for config, expected in [({}, 3600), ({'OCR_HTTP_TIMEOUT_SECONDS': '1200'}, 1200)]:
            with patch.dict(os.environ, config, clear=True), patch.object(httpx, 'AsyncClient', return_value=client) as constructor:
                self.assertEqual(await ns['run_http_ocr']('http://unused', 'test.pdf', b'pdf', 'test'), {'full_text': 'ok'})
                timeout = constructor.call_args.kwargs['timeout']
                self.assertEqual(timeout.read, expected)
                self.assertEqual(timeout.connect, 10)

    async def test_timeout_never_starts_fallback(self):
        ns = self.namespace()
        ns['run_http_ocr'] = AsyncMock(side_effect=httpx.ReadTimeout('slow PDF'))
        ns['run_legacy_ocr'] = AsyncMock()
        exec(compile(ast.Module(body=[work], type_ignores=[]), '<test>', 'exec'), ns)
        with self.assertRaises(HTTPException) as error:
            await ns['do_ocr']()
        self.assertEqual(error.exception.status_code, 504)
        ns['run_legacy_ocr'].assert_not_called()

    async def test_backend_connection_error_is_actionable(self):
        ns = self.namespace()
        ns['run_http_ocr'] = AsyncMock(side_effect=httpx.ConnectError('connection refused'))
        ns['run_legacy_ocr'] = AsyncMock()
        exec(compile(ast.Module(body=[work], type_ignores=[]), '<test>', 'exec'), ns)
        with self.assertRaises(HTTPException) as error:
            await ns['do_ocr']()
        self.assertEqual(error.exception.status_code, 502)
        self.assertIn('ConnectError', error.exception.detail)
        ns['run_legacy_ocr'].assert_not_called()

    async def test_backend_http_status_is_preserved(self):
        ns = self.namespace()
        ns['run_http_ocr'] = AsyncMock(side_effect=HTTPException(503, 'GPU busy'))
        exec(compile(ast.Module(body=[work], type_ignores=[]), '<test>', 'exec'), ns)
        with self.assertRaises(HTTPException) as error:
            await ns['do_ocr']()
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(error.exception.detail, 'GPU busy')
        ns['run_http_ocr'].assert_awaited_once()

if __name__ == '__main__':
    unittest.main()
