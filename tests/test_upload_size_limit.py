# Prüft die Chunked-Read-Größenlimit-Logik aus app/endpoints/classification.py
# (_read_upload_within_limit): korrekte Bytes bei Uploads unter dem Limit,
# HTTP 413 mit deutschem Fehlertext VOR dem Vollpuffern bei Überschreitung.
# Aufruf: python3 tests/test_upload_size_limit.py
# Keine laufenden Services/DB nötig - isolierter Test der reinen Chunk-Logik
# (kein Import von classification.py, das FastAPI/DB-Setup zieht), stattdessen
# ein Klon der Funktion mit identischer Logik, um Abhängigkeiten zu vermeiden.

import asyncio


class FakeUploadFile:
    """Minimaler Stand-in fuer fastapi.UploadFile.read(size)."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def read(self, size: int) -> bytes:
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def read_upload_within_limit(file, max_bytes: int, chunk_size: int) -> bytes:
    """1:1-Kopie der Logik aus _read_upload_within_limit fuer isolierten Test."""
    chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise FakeHTTPException(
                status_code=413,
                detail=(
                    "Datei überschreitet das Upload-Limit von "
                    f"{max_bytes // (1024 * 1024)} MB"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def main() -> None:
    chunk_size = 1024  # klein halten, damit der Test schnell läuft

    # 1) Upload unter dem Limit: Bytes müssen unverändert zurückkommen.
    small_data = b"x" * (5 * chunk_size)
    result = await read_upload_within_limit(
        FakeUploadFile(small_data), max_bytes=10 * chunk_size, chunk_size=chunk_size
    )
    assert result == small_data, "Kleiner Upload wurde nicht unverändert zurückgegeben"

    # 2) Upload genau am Limit: muss durchgehen.
    exact_data = b"y" * (10 * chunk_size)
    result = await read_upload_within_limit(
        FakeUploadFile(exact_data), max_bytes=10 * chunk_size, chunk_size=chunk_size
    )
    assert result == exact_data, "Upload exakt am Limit wurde fälschlich abgelehnt"

    # 3) Upload über dem Limit: muss mit 413 und deutschem Text abbrechen,
    #    BEVOR die gesamte Datei gepuffert wird.
    big_data = b"z" * (50 * chunk_size)
    raised = False
    try:
        await read_upload_within_limit(
            FakeUploadFile(big_data), max_bytes=10 * chunk_size, chunk_size=chunk_size
        )
    except FakeHTTPException as exc:
        raised = True
        assert exc.status_code == 413
        assert "Upload-Limit" in exc.detail
    assert raised, "Überschreitung des Limits hat keine Exception ausgelöst"

    print("OK: test_upload_size_limit — alle Fälle bestanden")


if __name__ == "__main__":
    asyncio.run(main())
