"""Prüft die Chunked-Read-Größenlimit-Logik DIREKT an
endpoints.classification._read_upload_within_limit.

Konsolidierung (Sweep-Follow-up 8, 2026-07-13): vorher lebte hier ein
1:1-Klon der Funktion "um FastAPI/DB-Setup zu vermeiden"; conftest.py stellt
die App-Imports inzwischen bereit. Limit und Chunk-Größe sind Modul-
Konstanten und werden per monkeypatch klein gestellt.

Run: .venv/bin/python -m pytest tests/test_upload_size_limit.py -q
"""
import asyncio

import pytest
from fastapi import HTTPException

import endpoints.classification as classification

CHUNK = 1024  # klein halten, damit der Test schnell läuft


class FakeUploadFile:
    """Minimaler Stand-in fuer fastapi.UploadFile.read(size)."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def read(self, size: int) -> bytes:
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def _read(data: bytes, max_bytes: int, monkeypatch) -> bytes:
    monkeypatch.setattr(classification, "MAX_UPLOAD_BYTES", max_bytes)
    monkeypatch.setattr(classification, "UPLOAD_READ_CHUNK_SIZE", CHUNK)
    return asyncio.run(classification._read_upload_within_limit(FakeUploadFile(data)))


def test_upload_below_limit_returns_bytes_unchanged(monkeypatch):
    small_data = b"x" * (5 * CHUNK)
    assert _read(small_data, max_bytes=10 * CHUNK, monkeypatch=monkeypatch) == small_data


def test_upload_exactly_at_limit_passes(monkeypatch):
    exact_data = b"y" * (10 * CHUNK)
    assert _read(exact_data, max_bytes=10 * CHUNK, monkeypatch=monkeypatch) == exact_data


def test_upload_over_limit_raises_413_before_buffering(monkeypatch):
    big_data = b"z" * (50 * CHUNK)
    with pytest.raises(HTTPException) as exc_info:
        _read(big_data, max_bytes=10 * CHUNK, monkeypatch=monkeypatch)
    assert exc_info.value.status_code == 413
    assert "Upload-Limit" in exc_info.value.detail
